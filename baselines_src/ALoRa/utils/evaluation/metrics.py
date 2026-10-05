
from .basic_metrics import basic_metricor, generate_curve
import sys, getopt
import time, datetime
from typing import Callable
import math


# To store a single anomaly
class Term:
    def __init__(self, first, last, name):
        self._first_timestamp = first
        self._last_timestamp = last
        self._name = name

    def set_time(self, first, last):
        self._first_timestamp = first
        self._last_timestamp = last

    def get_time(self):
        return self._first_timestamp, self._last_timestamp

    def set_name(self, str):
        self._name = str

    def get_name(self):
        return self._name

    def __eq__(self, other):
        return self._first_timestamp == other.get_time()[0] and self._last_timestamp == other.get_time()[1]


class TaPR:
    def __init__(self, label, theta, delta):
        self._predictions = []  # list of Terms
        self._anomalies = []    # list of Terms
        self._ambiguous_inst = [] # list of Terms

        self._set_predictions = False
        self._set_anomalies = False

        assert(len(label) == 2)
        self._normal_lbl = label[0]
        self._anomal_lbl = label[1]

        self._theta = theta
        self._delta = delta
        pass

    def load_predictions(self, filename):
        ntoken = self._check_file_format(filename)

        if ntoken == 1:
            self._predictions = self._load_timeseries_file(filename)
        else:
            self._predictions = self._load_range_file(filename)
        self._set_prediction = True


    def load_anomalies(self, filename):
        ntoken = self._check_file_format(filename)

        if ntoken == 1:
            self._anomalies = self._load_timeseries_file(filename)
        else:
            self._anomalies = self._load_range_file(filename)
        self._set_anomalies = True

        self._gen_ambiguous()


    def _gen_ambiguous(self):
        for i in range(len(self._anomalies)):
            start_id = self._anomalies[i].get_time()[1] + 1
            end_id = start_id + self._delta -1

            #if the next anomaly occurs during the theta, update the end_id
            if i+1 < len(self._anomalies) and end_id > self._anomalies[i+1].get_time()[0]:
                end_id = self._anomalies[i+1].get_time()[0]

            self._ambiguous_inst.append(Term(start_id, end_id, str(i)))


    def _check_file_format(self, filename):
        # check the file's format
        f = open(filename, 'r', encoding='utf-8', newline='')
        line = f.readline()
        token = line.strip().split(',')
        f.close()
        return len(token)

    def _load_range_file(self, filename):
        temp_list = []
        f = open(filename, 'r', encoding='utf-8', newline='')
        for line in f.readlines():
            items = line.strip().split(',')
            if len(items) > 2:
                temp_list.append(Term(int(items[0]), int(items[1]), str(items[2])))
            else:
                temp_list.append(Term(int(items[0]), int(items[1]), 'undefined'))
        f.close()
        return temp_list

    def _load_timeseries_file(self, filename):
        return_list = []
        start_id = -1
        id = 0
        range_id = 1
        #set prev_val as a value different to normal and anomalous labels
        prev_val = self._anomal_lbl-1
        if prev_val == self._normal_lbl:
            prev_val -= 1

        f = open(filename, 'r', encoding='utf-8', newline='')
        for line in f.readlines():
            val = int(line.strip().split()[0])

            if val == self._anomal_lbl and prev_val == self._normal_lbl:
                start_id = id
            elif val == self._normal_lbl and prev_val == self._anomal_lbl:
                return_list.append(Term(start_id, id - 1, str(range_id)))
                range_id += 1
                start_id = 0
            elif start_id == -1 and val == self._anomal_lbl:
                start_id = 0

            id += 1
            prev_val = val
        f.close()
        if start_id != 0:
            return_list.append(Term(start_id, id-1, str(range_id)))

        return return_list


    def get_n_predictions(self):
        return len(self._predictions)

    def get_n_anomalies(self):
        return len(self._anomalies)

    # return a value with the detected anomaly list
    def TaR_d(self) -> {float, list}:
        total_score = 0.0
        detected_anomalies = []
        for anomaly_id in range(len(self._anomalies)):
            anomaly = self._anomalies[anomaly_id]
            ambiguous = self._ambiguous_inst[anomaly_id]

            max_score = self._sum_of_func(anomaly.get_time()[0], anomaly.get_time()[1],
                                          anomaly.get_time()[0], anomaly.get_time()[1], self._uniform_func)

            score = 0.0
            for prediction in self._predictions:
                score += self._overlap_and_subsequent_score(anomaly, ambiguous, prediction)

            if min(1.0, score / max_score) > self._theta:
                total_score += 1.0
                detected_anomalies.append(anomaly)

        if len(self._anomalies) == 0:
            return 0.0, []
        else:
            return total_score / len(self._anomalies), detected_anomalies

    # return a value with the detected prediction lists
    def TaP_d(self) -> {float, list}:
        correct_predictions = []
        total_score = 0.0
        for prediction in self._predictions:
            max_score = prediction.get_time()[1] - prediction.get_time()[0] + 1

            score = 0.0
            for anomaly_id in range(len(self._anomalies)):
                anomaly = self._anomalies[anomaly_id]
                ambiguous = self._ambiguous_inst[anomaly_id]

                score += self._overlap_and_subsequent_score(anomaly, ambiguous, prediction)

            if (score/max_score) > self._theta:
                total_score += 1.0
                correct_predictions.append(prediction)

        if len(self._predictions) == 0:
            return 0.0, []
        else:
            return total_score / len(self._predictions), correct_predictions


    def _detect(self, src_range: Term, ranges: list, theta: int) -> bool:
        rest_len = src_range.get_time()[1] - src_range.get_time()[0] + 1
        for dst_range in ranges:
            len = self._overlapped_len(src_range, dst_range)
            if len != -1:
                rest_len -= len
        return (float)(rest_len) / (src_range.get_time()[1] - src_range.get_time()[0] + 1) <= (1.0 - theta)

    def _overlapped_len(self, range1: Term, range2: Term) -> int:
        detected_start = max(range1.get_time()[0], range2.get_time()[0])
        detected_end = min(range1.get_time()[1], range2.get_time()[1])

        if detected_end < detected_start:
            return 0
        else:
            return detected_end - detected_start + 1

    def _min_max_norm(self, value: int, org_min: int, org_max: int, new_min: int, new_max: int) -> float:
        return (float)(new_min) + (float)(value - org_min) * (new_max - new_min) / (org_max - org_min)

    def _decaying_func(self, val: float) -> float:
        assert (-6 <= val <= 6)
        return 1 / (1 + math.exp(val))

    def _ascending_func(self, val: float) -> float:
        assert (-6 <= val <= 6)
        return 1 / (1 + math.exp(val * -1))

    def _uniform_func(self, val: float) -> float:
        return 1.0

    def _sum_of_func(self, start_time: int, end_time: int, org_start: int, org_end: int,
                     func: Callable[[float], float]) -> float:
        val = 0.0
        for timestamp in range(start_time, end_time + 1):
            val += func(self._min_max_norm(timestamp, org_start, org_end, -6, 6))
        return val

    def _overlap_and_subsequent_score(self, anomaly: Term, ambiguous: Term, prediction: Term) -> float:
        score = 0.0

        detected_start = max(anomaly.get_time()[0], prediction.get_time()[0])
        detected_end = min(anomaly.get_time()[1], prediction.get_time()[1])

        score += self._sum_of_func(detected_start, detected_end,
                                   anomaly.get_time()[0], anomaly.get_time()[1], self._uniform_func)

        detected_start = max(ambiguous.get_time()[0], prediction.get_time()[0])
        detected_end = min(ambiguous.get_time()[1], prediction.get_time()[1])

        score += self._sum_of_func(detected_start, detected_end,
                                   ambiguous.get_time()[0], ambiguous.get_time()[1], self._decaying_func)

        return score

    def TaR_p(self) -> float:
        total_score = 0.0
        for anomaly_id in range(len(self._anomalies)):
            anomaly = self._anomalies[anomaly_id]
            ambiguous = self._ambiguous_inst[anomaly_id]

            max_score = self._sum_of_func(anomaly.get_time()[0], anomaly.get_time()[1],
                                          anomaly.get_time()[0], anomaly.get_time()[1], self._uniform_func)

            score = 0.0
            for prediction in self._predictions:
                score += self._overlap_and_subsequent_score(anomaly, ambiguous, prediction)

            total_score += min(1.0, score/max_score)

        if len(self._anomalies) == 0:
            return 0.0
        else:
            return total_score / len(self._anomalies)

    def TaP_p(self) -> float:
        total_score = 0.0
        for prediction in self._predictions:
            max_score = prediction.get_time()[1] - prediction.get_time()[0] + 1

            score = 0.0
            for anomaly_id in range(len(self._anomalies)):
                anomaly = self._anomalies[anomaly_id]
                ambiguous = self._ambiguous_inst[anomaly_id]

                score += self._overlap_and_subsequent_score(anomaly, ambiguous, prediction)

            total_score += score/max_score

        if len(self._predictions) == 0:
            return 0.0
        else:
            return total_score / len(self._predictions)


def main(argv):
    predict_file = ''
    anomaly_file = ''
    delta = 0
    theta = 0.5
    alpha = 0.5
    label = [0,1]
    print_detail = False

    try:
        opts, args = getopt.getopt(argv, "hpi:c:d:t:l:a:", ["input file=", "attack file=", "delta=(default:0)", "theta=(default:0.5)", "label=[normal,anomaly](default:1,-1)", "alpha=(default:0.5)"])
    except getopt.GetoptError:
        print('Getopt Error')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('evaluation.py -i <prediction_file> -c <anomaly_file> {-p} {-a} <alpha> {-t} <theta> {-d} <delta> {-l} <label>')
            buf = '''-i: File with predictions
-c: File with anomalies (i.e., correct answer)
-t: Parameter theta for detection scoring 
    Set as float value from 0 to 1
    Default = 0.5
-d: Parameter delta for subsequent scoring
    Set as zero or more larger integer value
    Defualt = 0
-l: Normal and anomaly labels
    Set as two integers separate by ','
    Default = 1,-1
-p: Enable printing the list of detected anomalies and correct predictions
    No need input values 
-a: Parameter alpha indicating weight for the detection score
    Default = 0.5
            '''
            print(buf)
            sys.exit()
        elif opt == '-p':
            print_detail = True
        elif opt in ("-i"):
            predict_file = arg
        elif opt in ("-c"):
            anomaly_file = arg
        elif opt in ("-d"):
            delta = int(arg)
        elif opt in ("-t"):
            theta = float(arg)
            assert (0.0 <= theta and theta <= 1.0)
        elif opt in ("-l"):
            label = str(arg).strip().split(',')
            label = [int(label[0]), int(label[1])]
        elif opt in ("-a"):
            alpha = float(arg)
            assert(0.0 <= alpha and alpha <= 1.0)

    if len(predict_file) == 0:
        print('Error: Input the prediction file after -i.')
        return

    if len(anomaly_file) == 0:
        print('Error: Input the anomaly file (ground truth) after -a.')
        return

    ev = TaPR(label, theta, delta)

    ev.load_anomalies(anomaly_file)
    ev.load_predictions(predict_file)


    tard_value, detected_list = ev.TaR_d()
    tarp_value = ev.TaR_p()

    print('\n[TaR]:',  "%0.5f"%(alpha*tard_value + (1-alpha)*tarp_value))
    print("\t* Detection score:", "%0.5f"%tard_value)
    if print_detail:
        buf = '\t\tdetected anomalies: '
        if len(detected_list) == 0:
            buf += "None  "
        else:
            for value in detected_list:
                buf += value.get_name() + '(' + str(value.get_time()[0]) + ':' + str(value.get_time()[1]) + '), '
        print(buf[:-2])
    print("\t* Portion score:", "%0.5f"%tarp_value, "\n")


    tapd_value, correct_list = ev.TaP_d()
    tapp_value = ev.TaP_p()
    print('[TaP]:', "%0.5f"%(alpha*tapd_value + (1-alpha)*tapp_value))

    print("\t* Detection score:", "%0.5f"%tapd_value)
    if print_detail:
        buf = '\t\tcorrect predictions: '
        if len(correct_list) == 0:
            buf += "None  "
        else:
            for value in correct_list:
                buf += value.get_name() + '(' + str(value.get_time()[0]) + ':' + str(value.get_time()[1]) + '), '
        print(buf[:-2])
    print("\t* Portion score:", "%0.5f"%tapp_value, "\n")



if __name__ == '__main__':
    main(sys.argv[1:])
    
def get_metrics(score, labels, slidingWindow=100, pred=None, version='opt', thre=250):
    metrics = {}

    '''
    Threshold Independent
    '''
    grader = basic_metricor()
    # AUC_ROC, Precision, Recall, PointF1, PointF1PA, Rrecall, ExistenceReward, OverlapReward, Rprecision, RF, Precision_at_k = grader.metric_new(labels, score, pred, plot_ROC=False)
    AUC_ROC = grader.metric_ROC(labels, score)
    AUC_PR = grader.metric_PR(labels, score)

    # R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
    _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(labels.astype(int), score, slidingWindow, version, thre)


    '''
    Threshold Dependent
    if pred is None --> use the oracle threshold
    '''

    PointF1 = grader.metric_PointF1(labels, score, preds=pred)
    PointF1PA = grader.metric_PointF1PA(labels, score, preds=pred)
    EventF1PA = grader.metric_EventF1PA(labels, score, preds=pred)
    RF1 = grader.metric_RF1(labels, score, preds=pred)
    Affiliation_F = grader.metric_Affiliation(labels, score, preds=pred)

    metrics['AUC-PR'] = AUC_PR
    metrics['AUC-ROC'] = AUC_ROC
    metrics['VUS-PR'] = VUS_PR
    metrics['VUS-ROC'] = VUS_ROC

    metrics['Standard-F1'] = PointF1
    metrics['PA-F1'] = PointF1PA
    metrics['Event-based-F1'] = EventF1PA
    metrics['R-based-F1'] = RF1
    metrics['Affiliation-F'] = Affiliation_F
    return metrics


def get_metrics_pred(score, labels, pred, slidingWindow=100):
    metrics = {}

    grader = basic_metricor()

    PointF1 = grader.metric_PointF1(labels, score, preds=pred)
    PointF1PA = grader.metric_PointF1PA(labels, score, preds=pred)
    EventF1PA = grader.metric_EventF1PA(labels, score, preds=pred)
    RF1 = grader.metric_RF1(labels, score, preds=pred)
    Affiliation_F = grader.metric_Affiliation(labels, score, preds=pred)
    VUS_R, VUS_P, VUS_F = grader.metric_VUS_pred(labels, preds=pred, windowSize=slidingWindow)

    metrics['Standard-F1'] = PointF1
    metrics['PA-F1'] = PointF1PA
    metrics['Event-based-F1'] = EventF1PA
    metrics['R-based-F1'] = RF1
    metrics['Affiliation-F'] = Affiliation_F

    metrics['VUS-Recall'] = VUS_R
    metrics['VUS-Precision'] = VUS_P
    metrics['VUS-F'] = VUS_F

    return metrics


def run_tapr(pred_vector, gt_vector, theta=0.5, delta=50, labels=(0, 1), verbose=False):
    """
    pred_vector: binary numpy array of model predictions (0/1)
    gt_vector: binary numpy array of ground truth labels (0/1)
    """
    assert len(pred_vector) == len(gt_vector), "Prediction and ground truth lengths do not match."

    # Save temporary files
    with open("temp_preds.txt", "w") as f:
        for val in pred_vector:
            f.write(f"{int(val)}\n")

    with open("temp_gt.txt", "w") as f:
        for val in gt_vector:
            f.write(f"{int(val)}\n")
    ev = TaPR(label=labels, theta=theta, delta=delta)
    ev.load_anomalies("temp_gt.txt")
    ev.load_predictions("temp_preds.txt")

    tard_value, detected_list = ev.TaR_d()
    tarp_value = ev.TaR_p()
    tapd_value, correct_list = ev.TaP_d()
    tapp_value = ev.TaP_p()

    tar_final = theta * tard_value + (1 - theta) * tarp_value
    tap_final = theta * tapd_value + (1 - theta) * tapp_value

    if verbose:
        print(f"[TaR]: {tar_final:.4f} (Detection: {tard_value:.4f}, Portion: {tarp_value:.4f})")
        print(f"[TaP]: {tap_final:.4f} (Detection: {tapd_value:.4f}, Portion: {tapp_value:.4f})")

    return {
        "TaR": tar_final,
        "TaR_detection": tard_value,
        "TaR_portion": tarp_value,
        "TaP": tap_final,
        "TaP_detection": tapd_value,
        "TaP_portion": tapp_value
    }
