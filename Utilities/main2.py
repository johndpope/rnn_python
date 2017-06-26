from RNN import music_info
from Utilities import music_utilities

music_info = music_info.MusicInfo("/home/bristina/PycharmProjects/seq2seq/proto/proto_database")
music_info.write_file_info()
# print(len(music_info.class_dictionary))

music_info = music_utilities.read_database_info()
print(music_info[1])

# ok = True
#
# for k in dict_class.keys():
#     if dict_class[k] != music_info.class_dictionary[k]:
#         ok = False
#
# print(ok)








