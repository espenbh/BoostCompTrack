def get_salmon_ID_and_comp_type_from_comp_ID(comp_ID, num_comp = 9):
    ''' 
    Get salmon ID and body salmon component type from body part ID. Assume the body part IDs start on 1.
    
    Args:
        bp_ID (int): Body part ID.
        num_comp (int): Number of salmon components.
    Return
        Salmon component (int)
        Samon ID (int)

    '''
    
    salmon_comp_int = (comp_ID)%num_comp
    salmon_ID = (comp_ID)//num_comp
    return salmon_ID, salmon_comp_int


def get_comp_id_from_salmon_ID_and_comp_type(salmon_id, comp_type, num_comp = 9):
    '''
    Get bodypart id.

    Args:
        id (int): id of the salmon
        comp_type (int): component of the salmon
    Return:
        bp_id (int): A unique id for the salmon component
    '''
    return salmon_id*num_comp + comp_type