#from car_damage.postprocess.vin.vin_claim import *
import os,json
import numpy as np

def post_process(vin, results):
    current_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_path, "../", "vin/external_api/case_vin.json")) as f:
        caseId_vin = json.load(f)

    child_folders = get_all_child_folder_name(results)
    out_put_list = {}
    out_dict = {}
    for child_folder in child_folders:
        out_dict = damage_info_from_forder(child_folder, results, caseId_vin, vin)
        out_put_list = {**out_put_list, **out_dict}

    out_dict=filter_loose(out_dict)
    return out_dict

def get_all_file(folder_path):
    files = []
    for r, d, f in os.walk(folder_path):
        for file in f:
            if 'mileage' in file or 'vin_number' in file or 'vehicle_registration' in file :
                continue
            if ('.jpg' in file.lower()) or ('.jpeg' in file.lower() or ('.png' in file.lower())):
                files.append(os.path.join(r, file))
    return files

def trans_json_result(vin, request_id, results):
    final_result = {"make": "N/A", "model": "N/A", "pic_url": "/pic/1571068279556.JPEG", "quote": "\u20ac3076",
                    "vin": "undefined", "year": "N/A"}
    quote = 0
    final_result["items"] = []
    if request_id not in results:
        return final_result

    for result in results[request_id]:
        result = result['parts']

        for damage in result:
            tmp = {}
            carpart_side = damage['part'].split("+")
            quote += get_quote(carpart_side[0], damage['treatment'])
            tmp["carpart"] = carpart_side[0]
            side=""
            if len(carpart_side)>1:
                side=carpart_side[1]

            if (len(side) == 2):
                tmp["Side 1"] = "Front" if 'f' in carpart_side[1] else "Rear"
                tmp["Side 2"] = "Left" if 'l' in carpart_side[1] else "Right"
            elif('f' in side or 'b' in side):
                tmp["Side 1"] = "Front" if 'f' in carpart_side[1] else "Rear"
                tmp["Side 2"] = "N/A"
            elif ('l' in side or 'r' in side):
                tmp["Side 1"] = "N/A"
                tmp["Side 2"] = "Left" if 'l' in carpart_side[1] else "Right"
            else:
                tmp["Side 1"] = "N/A"
                tmp["Side 2"] = "N/A"

            tmp["confidence"] = [round(confi['confidence'], 2) for confi in damage['damage']]
            tmp["damage"] = []

            tmp["damage"] = [[type['type'].replace('totaled', 'severe'),type['damage_number']] for type in damage['damage']]
            tmp["treatment"] = damage['treatment']
            final_result["items"].append(tmp)

    # if (vin == "undefined"):
    #     #     final_result["quote"] = "\u20ac0" + str(quote)
    fixed_car_lookup = get_car_from_vin(vin)

    final_result["year"] = fixed_car_lookup["year"]
    final_result["model"] = fixed_car_lookup["model"]
    final_result["make"] = fixed_car_lookup["make"]
    if final_result["make"] == "N/A":
        final_result["quote"] = "N/A"
    else:
        final_result["quote"] = "\u20ac" + str(quote)

    return final_result

def filter_loose(final_result):
    for uuid,results in final_result.items():
        for i,result in enumerate (results):
            parts  =result['parts']
            for j in range(len(parts)-1,-1,-1) :
                damage = parts[j]['damage']
                if len(damage) ==1 and 'loose' in damage[0]['type']: 
                    final_result[uuid][i]['parts'].pop(j)
                    
    return final_result


def is_eligible_damage_on_part(carpart, damage):
    # print(carpart, damage)
    if "grille" in carpart and damage in ["scratch", "dent"]:
        return False
    if "light" in carpart and "bezel" not in carpart and "dent" in damage:
        return False
    if "handle" in carpart and damage in ["dent"]:
        return False
    if ("windshield" in carpart  or 'window' in carpart) and damage in ["dent"]:
        return False
    return True

def getAngle(list_points):
    a, b, c = list_points
    ba = a - b
    bc = c - b
    if (np.linalg.norm(ba) * np.linalg.norm(bc) == 0):
        return 1.57
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(min(cosine_angle, 0.99))
    return angle

def check_carpart_in_list(carpart,target):

    for t in target:
        if carpart in t:
            return True
    return False 

