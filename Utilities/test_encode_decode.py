from RNN import music_info
from Utilities import music_utilities

# music_info = music_info.MusicInfo("/home/bristina/PycharmProjects/seq2seq/proto/proto_database")
# music_info.write_file_info()

music_inf = music_utilities.read_database_info()
print(music_inf[0])
print(music_inf[1])

