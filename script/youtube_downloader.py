import pytube


video_list = {
    'twice': [
        'https://www.youtube.com/watch?v=P9L5QCed3ao',
        'https://www.youtube.com/watch?v=Exp_REHm2Jw',
        'https://www.youtube.com/watch?v=XHaAQBCz4R8',
        'https://www.youtube.com/watch?v=3B4igIQ7MPQ',
        'https://www.youtube.com/watch?v=byjpaGbGsNQ'],
    'redvelvet': [
        'https://www.youtube.com/watch?v=EuMN_xUR5Uw',
        'https://www.youtube.com/watch?v=CATzR4PvCJo',
        'https://www.youtube.com/watch?v=udIG6pE8Grk',
        'https://www.youtube.com/watch?v=UZncgkoK5AU',
        'https://www.youtube.com/watch?v=6FkryVN5J_c'],
    'BTS': [
        'https://www.youtube.com/watch?v=rh7aKl0djIk',
        'https://www.youtube.com/watch?v=dBDEu4eMSv0',
        'https://www.youtube.com/watch?v=E-K8mB15KGI',
        'https://www.youtube.com/watch?v=_UagXHUrroU',
        'https://www.youtube.com/watch?v=NGyCUnrIS74'],
    'momoland': [
        'https://www.youtube.com/watch?v=7qT9YrPPGes',
        'https://www.youtube.com/watch?v=j_7pxzWmypA',
        'https://www.youtube.com/watch?v=J7g0ogbA9ZU',
        'https://www.youtube.com/watch?v=7qT9YrPPGes',
        'https://www.youtube.com/watch?v=F4PXsyelj-o']
}

for k, fl in video_list.items():
    for f_i, url in enumerate(fl):
        file_prefix = "%s_%d" % (k, f_i)
        pytube.YouTube(url).streams.filter(subtype='mp4').filter(res='720p') \
              .first().download(filename_prefix=file_prefix)
