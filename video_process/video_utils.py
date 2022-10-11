import cv2
# import torch
import math as m


def filter_carpart_by_view(info):
    def check_wrong_carpart_view(carpart_label, view):
        if carpart_label == 'fbu_front_bumper+f' and (view > 90 and view < 270):
            return False

        if carpart_label == 'rbu_rear_bumper+b' and (view < 90 or view > 270):
            return False

        if carpart_label == 'tail_gate+b' and (view < 135 or view > 225):
            return False

        if carpart_label in ['door+lf', 'door+lb'] and (view < 180):
            return False

        if carpart_label in ['door+rf', 'door+rb'] and (view > 180):
            return False

        return True

    for k in list(info['damage_result']):
        if not check_wrong_carpart_view(k, info['view']):
            del info['damage_result'][k]


def collect_final_result(damaged_by_bin_json):
    confirm = {}
    for k, value in damaged_by_bin_json.items():
        #     print(value)
        for cp, damages in value.items():
            for d in damages:
                label = cp+'_'+d[0]
                if label not in confirm:
                    confirm[label] = 1
                else:
                    confirm[label] += 1

    return confirm


def collect_final_result_after_cross_check(damaged_by_bin_json):
    confirm = {}
    # confirm2 = {}
    for id, (k, value) in enumerate(damaged_by_bin_json.items()):
        #     print(value)
        for cp, damages in value.items():
            for d in damages:
                if d[0] == 'scratch':
                    if d[1] > 0.61 and (d[-1] or any([i in cp for i in ['mirror', 'rocker_panel', 'hood']])):
                        label = cp+'_'+d[0]
                        if label not in confirm:
                            confirm[label] = 1
                        else:
                            confirm[label] += 1

                    # if d[1] > 0.61 :
                    #     label = cp+'_'+d[0]
                    #     if label not in confirm2:
                    #         confirm2[label] = 1
                    #     else:
                    #         confirm2[label] += 1
                else:
                    # if id % 3 == 0:
                    #     continue
                    label = cp+'_'+d[0]
                    if label not in confirm:
                        confirm[label] = 1
                    else:
                        confirm[label] += 1

    # print('uncheck result  : ',len(confirm2.values()),confirm2)

    for k in list(confirm.keys()):
        if 'rocker_panel' in k and confirm[k] == 1:
            del confirm[k]
            continue

        damage = k[k.rfind('_')+1:]
        if damage == 'dent' and confirm[k] == 1:
            del confirm[k]

    return confirm


def outlier(a, b, c):
    a1 = m.atan(1/(b-a)) / m.pi*180
    a2 = m.atan(1/(b-c)) / m.pi*180

    if a1 < 0:
        a1 = 180 + a1
    if a2 < 0:
        a2 = 180 + a2

    angle = (a1+a2)

    check = False
    if angle < 3:
        check = True
        if a < 5 or c < 5:
            check = False

    if angle > 357:
        check = True
        if b < 10:
            check = False

    return check


def clean_outlier(pred_json, bin_length):
    angles = [int(k)*bin_length for k in pred_json.keys()]

    for i in range(len(angles)):
        if i == 0 or i == len(angles)-1:
            continue

        if outlier(angles[i-1], angles[i], angles[i+1]):
            pred_json.pop(str(angles[i]//bin_length), None)

    return pred_json


def draw_icon(image, start_angle, end_angle):
    center = (419, 284)
    radius = 252

    # if start_angle > 180 :
    #     start_angle = start_angle - 360

    # if end_angle > 180:
    #     end_angle = end_angle - 360

    # if start_angle < -90 :
    #     start_angle = start_angle + 360

    # if end_angle < -90:
    #     end_angle = end_angle + 360

    if abs(end_angle - start_angle) > 300:
        #end_angle = end_angle + 360
        ns = min(start_angle, end_angle)
        ne = max(start_angle, end_angle)
        ns = ns + 360

        start_angle = ns
        end_angle = ne

    # print('debug loading angle : ',start_angle,end_angle,' | ',start_angle-90,end_angle-90)
    image = cv2.ellipse(image, center, (radius, radius), 0,
                        start_angle-90, end_angle-90, (255, 255, 0), -1)

    return image


def compare_masks(image, damage, pred_json):
    final_output = {}

    carpart_info = pred_json['carpart']
    if len(carpart_info['labels']) == 0 or (len(carpart_info['labels']) > 2 and ('c_90_90' in carpart_info['totaled_info']['view_img'] or 'c_90_270' in carpart_info['totaled_info']['view_img'])):
        return pred_json, final_output

    pred_json, final_output = damage.get_damage2carpart(
        image, pred_json, final_output)

    rm_part_list = [carpart for carpart, value in final_output.items() if (
        "tyre" in carpart or "alloy_wheel" in carpart or "handle" in carpart)]
    for rm_part in rm_part_list:
        if rm_part in final_output:
            del final_output[rm_part]

    return pred_json, final_output
