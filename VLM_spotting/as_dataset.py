#!/usr/bin/env python3
"""
在训练镜像里用的读取层:把 PVC 上的 tar 分片 + index.sqlite 当成"一个文件夹"来随机访问。

两种用法:
  1) 按路径随机取任意文件(等价于 open(folder/path)):
        store = TarStore("/data/Action_Spotting", "/data/Action_Spotting/index.sqlite")
        raw_bytes = store.read("finegym/clipA/000076.jpg")     # 直接 seek 读那段字节
        ann = store.read("TouchMoment/Annotations/xxx.json")   # 非图像文件同样可取

  2) 当 PyTorch Dataset,按 clip 取一段帧(动作检测常用):
        ds = ClipDataset("/data/Action_Spotting", "/data/Action_Spotting/index.sqlite",
                         transform=your_transform)
        sample = ds[0]          # {'clip': 'finegym/clipA', 'frames': [Tensor/PIL, ...]}
        loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=8,
                                             collate_fn=lambda b: b)   # 变长帧,自己 collate

设计要点:SQLite 只读连接 + shard 文件句柄都在 worker 进程里"懒加载",fork 安全;
随机读靠 seek(offset) 直取,不解包、不遍历散文件。
"""
import io, os, sqlite3

class TarStore:
    """按 offset 索引随机读取 tar 内文件的最小实现。"""
    def __init__(self, root, index_path):
        self.root = root
        self.index_path = index_path
        self._db = None
        self._fh = {}          # shard -> 打开的文件句柄(每个 worker 各自一份)

    def _conn(self):
        if self._db is None:
            self._db = sqlite3.connect(f"file:{self.index_path}?mode=ro",
                                       uri=True, check_same_thread=False)
        return self._db

    def _fh_of(self, shard):
        fh = self._fh.get(shard)
        if fh is None:
            fh = open(os.path.join(self.root, shard), "rb", buffering=0)
            self._fh[shard] = fh
        return fh

    def read(self, path):
        """返回 path 对应文件的原始字节(随机访问)。"""
        row = self._conn().execute(
            "SELECT shard, offset, size FROM files WHERE path=?", (path,)).fetchone()
        if row is None:
            raise KeyError(path)
        shard, off, size = row
        fh = self._fh_of(shard)
        fh.seek(off)
        return fh.read(size)

    def _read_at(self, shard, off, size):
        fh = self._fh_of(shard)
        fh.seek(off)
        return fh.read(size)

    def clips(self):
        return [r[0] for r in self._conn().execute(
            "SELECT DISTINCT clip FROM files WHERE clip IS NOT NULL ORDER BY clip")]

    def clip_frames(self, clip):
        """返回该 clip 的帧 [(path, shard, offset, size), ...],已按 ord 排序。"""
        return self._conn().execute(
            "SELECT path, shard, offset, size FROM files WHERE clip=? ORDER BY ord",
            (clip,)).fetchall()


try:
    import torch
    from PIL import Image

    class ClipDataset(torch.utils.data.Dataset):
        """每个样本 = 一个 clip 的全部帧(解码为 PIL/Tensor)。"""
        def __init__(self, root, index_path, transform=None):
            self.root, self.index_path, self.transform = root, index_path, transform
            # __init__ 里只短暂查一次 clip 列表就关闭,避免把连接带过 fork
            con = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
            self.clip_ids = [r[0] for r in con.execute(
                "SELECT DISTINCT clip FROM files WHERE clip IS NOT NULL ORDER BY clip")]
            con.close()
            self._store = None

        def _store_lazy(self):
            if self._store is None:            # 在各 worker 进程内首次访问时创建,fork 安全
                self._store = TarStore(self.root, self.index_path)
            return self._store

        def __len__(self):
            return len(self.clip_ids)

        def __getitem__(self, i):
            clip = self.clip_ids[i]
            store = self._store_lazy()
            frames = []
            for _path, shard, off, size in store.clip_frames(clip):
                img = Image.open(io.BytesIO(store._read_at(shard, off, size))).convert("RGB")
                frames.append(self.transform(img) if self.transform else img)
            return {"clip": clip, "frames": frames}
except ImportError:
    pass  # 没装 torch/PIL 时,TarStore 仍可单独使用


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "/data/Action_Spotting"
    idx = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "index.sqlite")
    store = TarStore(root, idx)
    clips = store.clips()
    print(f"clip 总数: {len(clips)}")
    if clips:
        c = clips[0]
        fr = store.clip_frames(c)
        print(f"示例 clip: {c}  帧数: {len(fr)}")
        first = fr[0]
        data = store._read_at(first[1], first[2], first[3])
        print(f"第一帧 {first[0]}  字节数: {len(data)}  (JPEG 头: {data[:3].hex()})")
