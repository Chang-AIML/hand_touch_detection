"""Vendored from hand_touch_detection/common/eval.py (verbatim logic).

Provides the temporal NMS post-processings used by eval_nms.py:
  non_maximum_supression(pred, window, threshold=0.0)      -> hard greedy NMS
  soft_non_maximum_supression(pred, window, threshold=0.01) -> parabolic soft-NMS

`pred` is the same per-video list-of-dicts format the scorer consumes:
  [{"video": vid, "events": [{"label", "frame", "score"}, ...]}, ...]
Each returns a NEW list (deep-copied) with suppressed `events` (+ "num_events").

Only third-party dep: numpy. The ErrorStat / ForegroundF1 / process_frame_predictions
helpers are carried over verbatim for completeness but are not needed by the scorer.
"""
import copy
from collections import defaultdict
import numpy as np


class ErrorStat:

    def __init__(self):
        self._total = 0
        self._err = 0

    def update(self, true, pred):
        self._err += np.sum(true != pred)
        self._total += true.shape[0]

    def get(self):
        return self._err / self._total

    def get_acc(self):
        return 1. - self._get()


class ForegroundF1:

    def __init__(self):
        self._tp = defaultdict(int)
        self._fp = defaultdict(int)
        self._fn = defaultdict(int)

    def update(self, true, pred):
        if pred != 0:
            if true != 0:
                self._tp[None] += 1
            else:
                self._fp[None] += 1

            if pred == true:
                self._tp[pred] += 1
            else:
                self._fp[pred] += 1
                if true != 0:
                     self._fn[true] += 1
        elif true != 0:
            self._fn[None] += 1
            self._fn[true] += 1

    def get(self, k):
        return self._f1(k)

    def tp_fp_fn(self, k):
        return self._tp[k], self._fp[k], self._fn[k]

    def _f1(self, k):
        denom = self._tp[k] + 0.5 * self._fp[k] + 0.5 * self._fn[k]
        if denom == 0:
            assert self._tp[k] == 0
            denom = 1
        return self._tp[k] / denom


def process_frame_predictions(
        dataset, classes, pred_dict, high_recall_score_threshold=0.01
):
    classes_inv = {v: k for k, v in classes.items()}

    fps_dict = {}
    for video, _, fps in dataset.videos:
        fps_dict[video] = fps

    err = ErrorStat()
    f1 = ForegroundF1()

    pred_events = []
    pred_events_high_recall = []
    pred_scores = {}
    for video, (scores, support) in sorted(pred_dict.items()):
        label = dataset.get_labels(video)
        # support[support == 0] = 1   # get rid of divide by zero
        assert np.min(support) > 0, (video, support.tolist())
        scores /= support[:, None]
        pred = np.argmax(scores, axis=1)
        err.update(label, pred)

        pred_scores[video] = scores.tolist()

        events = []
        events_high_recall = []
        for i in range(pred.shape[0]):
            f1.update(label[i], pred[i])

            if pred[i] != 0:
                events.append({
                    'label': classes_inv[pred[i]],
                    'frame': i,
                    'score': scores[i, pred[i]].item()
                })

            for j in classes_inv:
                if scores[i, j] >= high_recall_score_threshold:
                    events_high_recall.append({
                        'label': classes_inv[j],
                        'frame': i,
                        'score': scores[i, j].item()
                    })

        pred_events.append({
            'video': video, 'events': events,
            'fps': fps_dict[video]})
        pred_events_high_recall.append({
            'video': video, 'events': events_high_recall,
            'fps': fps_dict[video]})

    return err, f1, pred_events, pred_events_high_recall, pred_scores


def non_maximum_supression(pred, window, threshold=0.0):
    """Hard NMS (greedy), per video per label -- matches bbvisual/hice util/eval.py.

    Repeatedly takes the highest-scoring event, keeps it, and DELETES every
    same-label event within +-window frames of it; repeats until the pool is
    empty or the top score falls below `threshold`. Kept events are therefore
    at least `window`+1 frames apart. `window` may be a scalar or a per-label
    list."""
    preds = copy.deepcopy(pred)
    new_pred = []
    for video_pred in preds:
        events_by_label = defaultdict(list)
        for e in video_pred['events']:
            events_by_label[e['label']].append(e)

        events = []
        i = 0
        for v in events_by_label.values():
            class_window = window if not isinstance(window, list) else window[i]
            i += 1
            while len(v) > 0:
                e1 = max(v, key=lambda x: x['score'])
                if e1['score'] < threshold:
                    break
                pos1 = [p for p, e in enumerate(v) if e['frame'] == e1['frame']][0]
                events.append(copy.deepcopy(e1))
                v.pop(pos1)
                list_pos = [p for p, e in enumerate(v)
                            if (e['frame'] >= e1['frame'] - class_window)
                            and (e['frame'] <= e1['frame'] + class_window)]
                for p in list_pos[::-1]:
                    v.pop(p)
        events.sort(key=lambda x: x['frame'])
        new_video_pred = copy.deepcopy(video_pred)
        new_video_pred['events'] = events
        new_video_pred['num_events'] = len(events)
        new_pred.append(new_video_pred)
    return new_pred


def soft_non_maximum_supression(pred, window, threshold=0.01):
    """Soft-NMS (parabolic), per video per label -- matches bbvisual/hice util/eval.py.

    Repeatedly takes the highest-scoring event M and keeps it, then decays every
    same-label event within +-window of M by a parabolic penalty:
        score *= (|frame - M.frame|)^2 / window^2
    i.e. immediate neighbours (small distance) are strongly suppressed
    (penalty -> 0), window-edge neighbours are barely touched (penalty -> 1),
    and events beyond +-window are untouched. Stops when the top remaining
    score falls below `threshold`; the rest are dropped. `window` may be a
    scalar or a per-label list."""
    preds = copy.deepcopy(pred)
    new_pred = []
    for video_pred in preds:
        events_by_label = defaultdict(list)
        for e in video_pred['events']:
            events_by_label[e['label']].append(e)

        events = []
        i = 0
        for v in events_by_label.values():
            class_window = window if not isinstance(window, list) else window[i]
            i += 1
            while len(v) > 0:
                e1 = max(v, key=lambda x: x['score'])
                if e1['score'] < threshold:
                    break
                pos1 = [p for p, e in enumerate(v) if e['frame'] == e1['frame']][0]
                events.append(copy.deepcopy(e1))
                list_pos = [p for p, e in enumerate(v)
                            if (e['frame'] >= e1['frame'] - class_window)
                            and (e['frame'] <= e1['frame'] + class_window)]
                for p in list_pos:
                    v[p]['score'] = v[p]['score'] * \
                        (np.abs(e1['frame'] - v[p]['frame'])) ** 2 / (class_window ** 2)
                v.pop(pos1)
        events.sort(key=lambda x: x['frame'])
        new_video_pred = copy.deepcopy(video_pred)
        new_video_pred['events'] = events
        new_video_pred['num_events'] = len(events)
        new_pred.append(new_video_pred)
    return new_pred
