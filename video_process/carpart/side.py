import numpy as np
from numpy import linalg as LA
# list carpart left_right
verify_capart_left_right = ["fli_fog_light", "sli_side_turn_light", "mirror", "bpn_bed_panel", "mpa_mid_panel",
                            "fender", "hli_head_light", "door", "lli_low_bumper_tail_light", "tyre", "alloy_wheel",
                            "rocker_panel", "qpa_quarter_panel", "tli_tail_light", "fbe_fog_light_bezel", "window"]

verify_carpart_only_lr = ["rocker_panel"]
# list carpart front_back
verify_capart_front_back = ["door", "tyre", "alloy_wheel", "window"]

font_view = ['fli_fog_light', 'sli_side_turn_light', 'mirror', 'fbu_front_bumper', 'grille', 'fender', 'hli_head_light',
             'hood', 'hpa_header_panel', 'fbe_fog_light_bezel', 'fwi_windshield']
back_view = ['bpn_bed_panel', 'tail_gate', 'mpa_mid_panel', 'rbu_rear_bumper', 'lli_low_bumper_tail_light', 'trunk',
             'hsl_high_mount_stop_light', 'qpa_quarter_panel', 'rpa_rear_panel', 'tli_tail_light']

def get_view_score(center_capart, view_center_list):
    score = 0
    if (len(view_center_list) == 0):
        return score
    center_capart = np.asarray(center_capart)
    for view_center in view_center_list:
        view_center = np.asarray(view_center)
        score = score + np.exp(-LA.norm(center_capart - view_center))

    return score / len(view_center_list)

def get_side_carpart(real_carpart_center_list, view):
    
    if view is None:
        view=90
    m_real_carpart_center_list = [info[:-1] for info in real_carpart_center_list]                 
    # check left right
    for carpart_lr in verify_capart_left_right:

        index_list = [i for i, value in enumerate(m_real_carpart_center_list) if (value[0] == carpart_lr)]
        if (len(index_list) == 0):
            continue

        if (int(view) <= 180):
            nv_carpart_view = ["tli_tail_light", "qpa_quarter_panel", "lli_low_bumper_tail_light"]
            prior_label1 = "+r"
            prior_label2 = "+l"
        else:
            nv_carpart_view = ["hli_head_light", "fbe_fog_light_bezel", "mirror", "fli_fog_light"]
            prior_label1 = "+l"
            prior_label2 = "+r"

        if ((carpart_lr in verify_capart_front_back) or len(index_list) == 1):
            
            for index in index_list:
                
                #  check carpart by grille
                if (0 <= int(view)<=40  or int(view)>=320 ) and (carpart_lr in ["hli_head_light", "fbe_fog_light_bezel", "mirror", "fli_fog_light","tyre","alloy_wheel"]):
                    grille_info= [value[1] for i, value in enumerate(m_real_carpart_center_list) if ('grille' in value[0])]
                    
                    if len(grille_info) > 0:
                        if (m_real_carpart_center_list[index][1] - np.mean(grille_info)) <0 :
                            m_real_carpart_center_list[index][0] = m_real_carpart_center_list[index][0] + '+r'
                        else :
                            m_real_carpart_center_list[index][0] = m_real_carpart_center_list[index][0] + '+l'
                        continue
                    m_real_carpart_center_list[index][0] = m_real_carpart_center_list[index][0] + prior_label1
                    continue
                
                #  check carpart by fwi_windshield
                if (0 <= int(view)<=40  or int(view)>=320 ) and (carpart_lr in ["mirror"]):
                    grille_info= [value for i, value in enumerate(m_real_carpart_center_list) if ('fwi_windshield' in value[0])]
                    if len(grille_info) > 0:
                        if (m_real_carpart_center_list[index][1] - grille_info[0][1]) <0 :
                            m_real_carpart_center_list[index][0] = m_real_carpart_center_list[index][0] + '+r'
                        else :
                            m_real_carpart_center_list[index][0] = m_real_carpart_center_list[index][0] + '+l'
                        continue
                    m_real_carpart_center_list[index][0] = m_real_carpart_center_list[index][0] + prior_label1
                    continue
                
                # sometime tail_light is missing 
                if  carpart_lr in ["tli_tail_light"]:

                    quarter_panel_info= [value for i, value in enumerate(m_real_carpart_center_list) if ('qpa_quarter_panel' in value[0])]
                    
                    if (len(quarter_panel_info) ==1 and quarter_panel_info[0][1] > 0.6)  or len(quarter_panel_info) >1:
                
                        if (150 <=int(view)<=190  and   m_real_carpart_center_list[index][1] < 0.35):
                            m_real_carpart_center_list[index][0] = m_real_carpart_center_list[index][0] + '+l'
                            continue

                    if (len(quarter_panel_info) ==1 and quarter_panel_info[0][1] < 0.6)  or len(quarter_panel_info) >1:

                        if (170 <=int(view)<230  and  m_real_carpart_center_list[index][1] > 0.65):
                            m_real_carpart_center_list[index][0] = m_real_carpart_center_list[index][0] + '+r'
                            continue

                if (0 <= int(view)<=40  and  m_real_carpart_center_list[index][1] > 0.65):
                    m_real_carpart_center_list[index][0] = m_real_carpart_center_list[index][0] + prior_label2
                    continue 
                    # print('mmmmmmmmmmmmmmmmmprior_label2',view,carpart_lr,prior_label2)
                    

                m_real_carpart_center_list[index][0] = m_real_carpart_center_list[index][0] + prior_label1
        else:
            index1, index2 = index_list[:2]
            center_x1 = m_real_carpart_center_list[index1][1]
            center_x2 = m_real_carpart_center_list[index2][1]

            if carpart_lr in verify_carpart_only_lr:
                m_real_carpart_center_list[index1][0] = m_real_carpart_center_list[index1][0] + prior_label1
                m_real_carpart_center_list[index2][0] = m_real_carpart_center_list[index2][0] + prior_label1
                continue

            if ((center_x1 < center_x2 and (carpart_lr not in nv_carpart_view)) or (
                    center_x1 > center_x2 and (carpart_lr in nv_carpart_view))):

                m_real_carpart_center_list[index1][0] = m_real_carpart_center_list[index1][0] + prior_label1
                m_real_carpart_center_list[index2][0] = m_real_carpart_center_list[index2][0] + prior_label2
            else:

                m_real_carpart_center_list[index1][0] = m_real_carpart_center_list[index1][0] + prior_label2
                m_real_carpart_center_list[index2][0] = m_real_carpart_center_list[index2][0] + prior_label1

    font_center_list = [value[1:-1] for i, value in enumerate(real_carpart_center_list) if (value[0] in font_view)]

    back_center_list = [value[1:-1] for i, value in enumerate(real_carpart_center_list) if (value[0] in back_view)]

    # check font back
    for carpart_lr in verify_capart_front_back:
        index_list = [i for i, value in enumerate(real_carpart_center_list) if (value[0] == carpart_lr)]
        if (len(index_list) == 0):
            continue

        fb_score_list = []
        for index in index_list:
            center = m_real_carpart_center_list[index][1:]
            font_score = get_view_score(center, font_center_list)
            back_score = get_view_score(center, back_center_list)
            fb_score_list.append([index, font_score, back_score])

        if (len(index_list) == 1):

            index, font_score, back_score = fb_score_list[0]
            if int(view) <= 50 or  int(view) >= 320:
                font_score =font_score + 0.2
            if  140 <=int(view) <=220:
                back_score =back_score +0.2

            if (font_score > back_score):
                m_real_carpart_center_list[index][0] = m_real_carpart_center_list[index][0] + "f"
            else:
                m_real_carpart_center_list[index][0] = m_real_carpart_center_list[index][0] + "b"
        else:
            index1, font_score1, back_score1 = fb_score_list[0]
            index2, font_score2, back_score2 = fb_score_list[-1]
            index3, font_score3, back_score3 = fb_score_list[1]

            
            if font_score1 ==0 and font_score2 ==0 and font_score3==0 and back_score1==0 and back_score2==0 and back_score3==0:
                
                if int(view) < 180:
                    font_score1 =0.0001 if  m_real_carpart_center_list[index1][1] >  m_real_carpart_center_list[index2][1] else -0.0001
            
                if  int(view) > 180:
                    back_score1 =0.0001 if m_real_carpart_center_list[index1][1] >  m_real_carpart_center_list[index2][1] else -0.0001
            
            if (font_score1 > font_score2 or back_score1 < back_score2):
                m_real_carpart_center_list[index1][0] = m_real_carpart_center_list[index1][0] + "f"
                m_real_carpart_center_list[index2][0] = m_real_carpart_center_list[index2][0] + "b"
                if index2 != index3:
                    m_real_carpart_center_list[index3][0] = m_real_carpart_center_list[index3][0] + "f"
            else:
                m_real_carpart_center_list[index1][0] = m_real_carpart_center_list[index1][0] + "b"
                m_real_carpart_center_list[index2][0] = m_real_carpart_center_list[index2][0] + "f"
                if index2 != index3:
                    m_real_carpart_center_list[index3][0] = m_real_carpart_center_list[index3][0] + "b"

    # add  font back
    for carpart_lr in font_view:
        index_list = [i for i, value in enumerate(real_carpart_center_list) if (value[0] == carpart_lr)]
        if (len(index_list) == 0):
            continue

        for index in index_list:
            if "+" in m_real_carpart_center_list[index][0]:
                m_real_carpart_center_list[index][0] = m_real_carpart_center_list[index][0] + "f"
            else:
                m_real_carpart_center_list[index][0] = m_real_carpart_center_list[index][0] + "+f"

    for carpart_lr in back_view:
        index_list = [i for i, value in enumerate(real_carpart_center_list) if (value[0] == carpart_lr)]
        if (len(index_list) == 0):
            continue
        for index in index_list:
            if "+" in m_real_carpart_center_list[index][0]:
                m_real_carpart_center_list[index][0] = m_real_carpart_center_list[index][0] + "b"
            else:
                m_real_carpart_center_list[index][0] = m_real_carpart_center_list[index][0] + "+b"

    return m_real_carpart_center_list


