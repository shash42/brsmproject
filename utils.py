def get_session_list(df_array):
    session_list = []
    curr_session = []
    prev_session_id = ''
    for i in range(len(df_array)):
        session_id = df_array[i][0]
        if session_id == prev_session_id:
            curr_session.append(df_array[i])
        else:
            if len(curr_session) > 0:
                session_list.append(curr_session)
            curr_session = []
            curr_session.append(df_array[i])
        prev_session_id = session_id
    return session_list

def disallow_context_change(session_list):
    output = []
    for session in session_list:
        # remove session from session_list if context change occurs 
        change = False
        for track in session:
            if track[8] == 1:
                change = True
                break
        if not change:
            output.append(session)
    return output