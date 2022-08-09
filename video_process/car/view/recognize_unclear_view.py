

clear_views=[['hood','grille'],['roof'],['tail_gate','rbu_rear_bumper']]
unclear_views_3part=[['hli_head_light','fbu_front_bumper','fbe_fog_light_bezel'],['mirror','hood','window'],['qpa_quarter_panel','alloy_wheel','tyre'],\
    ['fender','alloy_wheel','tyre'],['fender','door','sli_side_turn_light'],['fbu_front_bumper', 'hood', 'hli_head_light'],['tyre', 'rocker_panel', 'door'],['door', 'rocker_panel', 'door'],['qpa_quarter_panel','door','rocker_panel'],['qpa_quarter_panel','door','tyre'],['fender','door','tyre']]

unclear_views_4part=[['rocker_panel','door','door','tyre'],['rocker_panel','door','alloy_wheel','tyre'],['tyre','alloy_wheel','hood','fender']]
unclear_views_5part=[['door','door','door','window','handle']]

def check_unclear_view(labels,scores):
    if len(labels) <3:
        for i in range(len(labels)-1,-1,-1):
            if scores[i] <0.9 :
                labels.pop(i) 
        for clear_view in clear_views:
            if len(clear_view) !=len(labels):
                continue
            
            if set(labels).issubset(clear_view):
                return False
        return True


    for unclear_view in unclear_views_3part:
        if len(unclear_view) !=len(labels):
            continue 
        if set(labels).issubset(unclear_view):
            return True 


    for unclear_view in unclear_views_3part+unclear_views_4part + unclear_views_5part:
        if len(unclear_view) !=len(labels):
            continue 
        same_carpart=[carpart for carpart in unclear_view if unclear_view.count(carpart)==labels.count(carpart)]
        if len(same_carpart)==len(unclear_view):
            return True 

    return False
