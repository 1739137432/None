import numpy as np
import scipy
import pickle


def load_PABDMH_data(prefix='data/output/relationship/VI_step_data_division'):
    in_file = open(prefix + '/0/0-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00
    in_file.close()
    in_file = open(prefix + '/0/0-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    adjlist010 = [line.strip() for line in in_file]
    adjlist010 = adjlist010
    in_file.close()
    in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    idx010 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/0/0-2-0.adjlist', 'r')
    adjlist020 = [line.strip() for line in in_file]
    adjlist020 = adjlist020
    in_file.close()
    in_file = open(prefix + '/0/0-2-0_idx.pickle', 'rb')
    idx020 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/0/0-3-0.adjlist', 'r')
    adjlist030 = [line.strip() for line in in_file]
    adjlist030 = adjlist030
    in_file.close()
    in_file = open(prefix + '/0/0-3-0_idx.pickle', 'rb')
    idx030 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/0/0-4-0.adjlist', 'r')
    adjlist040 = [line.strip() for line in in_file]
    adjlist040 = adjlist040
    in_file.close()
    in_file = open(prefix + '/0/0-4-0_idx.pickle', 'rb')
    idx040 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/0/0-1-1-0.adjlist', 'r')
    adjlist0110 = [line.strip() for line in in_file]
    adjlist0110 = adjlist0110
    in_file.close()
    in_file = open(prefix + '/0/0-1-1-0_idx.pickle', 'rb')
    idx0110 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/0/0-2-2-0.adjlist', 'r')
    adjlist0220 = [line.strip() for line in in_file]
    adjlist0220 = adjlist0220
    in_file.close()
    in_file = open(prefix + '/0/0-2-2-0_idx.pickle', 'rb')
    idx0220 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/0/0-3-3-0.adjlist', 'r')
    adjlist0330 = [line.strip() for line in in_file]
    adjlist0330 = adjlist0330
    in_file.close()
    in_file = open(prefix + '/0/0-3-3-0_idx.pickle', 'rb')
    idx0330 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/0/0-4-4-0.adjlist', 'r')
    adjlist0440 = [line.strip() for line in in_file]
    adjlist0440 = adjlist0440
    in_file.close()
    in_file = open(prefix + '/0/0-4-4-0_idx.pickle', 'rb')
    idx0440 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/0/0-1-4-1-0.adjlist', 'r')
    adjlist01410 = [line.strip() for line in in_file]
    adjlist01410 = adjlist01410
    in_file.close()
    in_file = open(prefix + '/0/0-1-4-1-0_idx.pickle', 'rb')
    idx01410 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/0/0-2-4-2-0.adjlist', 'r')
    adjlist02420 = [line.strip() for line in in_file]
    adjlist02420 = adjlist02420
    in_file.close()
    in_file = open(prefix + '/0/0-2-4-2-0_idx.pickle', 'rb')
    idx02420 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/0/0-3-4-3-0.adjlist', 'r')
    adjlist03430 = [line.strip() for line in in_file]
    adjlist03430 = adjlist03430
    in_file.close()
    in_file = open(prefix + '/0/0-3-4-3-0_idx.pickle', 'rb')
    idx03430 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/0/0-4-1-4-0.adjlist', 'r')
    adjlist04140 = [line.strip() for line in in_file]
    adjlist04140 = adjlist04140
    in_file.close()
    in_file = open(prefix + '/0/0-4-1-4-0_idx.pickle', 'rb')
    idx04140 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/0/0-4-2-4-0.adjlist', 'r')
    adjlist04240 = [line.strip() for line in in_file]
    adjlist04240 = adjlist04240
    in_file.close()
    in_file = open(prefix + '/0/0-4-2-4-0_idx.pickle', 'rb')
    idx04240 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/0/0-4-3-4-0.adjlist', 'r')
    adjlist04340 = [line.strip() for line in in_file]
    adjlist04340 = adjlist04340
    in_file.close()
    in_file = open(prefix + '/0/0-4-3-4-0_idx.pickle', 'rb')
    idx04340 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/0/0-1-4-4-1-0.adjlist', 'r')
    adjlist014410 = [line.strip() for line in in_file]
    adjlist014410 = adjlist014410
    in_file.close()
    in_file = open(prefix + '/0/0-1-4-4-1-0_idx.pickle', 'rb')
    idx014410 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/0/0-2-4-4-2-0.adjlist', 'r')
    adjlist024420 = [line.strip() for line in in_file]
    adjlist024420 = adjlist024420
    in_file.close()
    in_file = open(prefix + '/0/0-2-4-4-2-0_idx.pickle', 'rb')
    idx024420 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/0/0-3-4-4-3-0.adjlist', 'r')
    adjlist034430 = [line.strip() for line in in_file]
    adjlist034430 = adjlist034430
    in_file.close()
    in_file = open(prefix + '/0/0-3-4-4-3-0_idx.pickle', 'rb')
    idx034430 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/0/0-4-1-1-4-0.adjlist', 'r')
    adjlist041140 = [line.strip() for line in in_file]
    adjlist041140 = adjlist041140
    in_file.close()
    in_file = open(prefix + '/0/0-4-1-1-4-0_idx.pickle', 'rb')
    idx041140 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/0/0-4-2-2-4-0.adjlist', 'r')
    adjlist042240 = [line.strip() for line in in_file]
    adjlist042240 = adjlist042240
    in_file.close()
    in_file = open(prefix + '/0/0-4-2-2-4-0_idx.pickle', 'rb')
    idx042240 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/0/0-4-3-3-4-0.adjlist', 'r')
    adjlist043340 = [line.strip() for line in in_file]
    adjlist043340 = adjlist043340
    in_file.close()
    in_file = open(prefix + '/0/0-4-3-3-4-0_idx.pickle', 'rb')
    idx043340 = pickle.load(in_file)
    in_file.close()
    # ====================================================================
    in_file = open(prefix + '/1/1-1.adjlist', 'r')
    adjlist11 = [line.strip() for line in in_file]
    adjlist11 = adjlist11
    in_file.close()
    in_file = open(prefix + '/1/1-1_idx.pickle', 'rb')
    idx11 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/1/1-0-1.adjlist', 'r')
    adjlist101 = [line.strip() for line in in_file]
    adjlist101 = adjlist101
    in_file.close()
    in_file = open(prefix + '/1/1-0-1_idx.pickle', 'rb')
    idx101 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/1/1-4-1.adjlist', 'r')
    adjlist141 = [line.strip() for line in in_file]
    adjlist141 = adjlist141
    in_file.close()
    in_file = open(prefix + '/1/1-4-1_idx.pickle', 'rb')
    idx141 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/1/1-0-0-1.adjlist', 'r')
    adjlist1001 = [line.strip() for line in in_file]
    adjlist1001 = adjlist1001
    in_file.close()
    in_file = open(prefix + '/1/1-0-0-1_idx.pickle', 'rb')
    idx1001 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/1/1-4-4-1.adjlist', 'r')
    adjlist1441 = [line.strip() for line in in_file]
    adjlist1441 = adjlist1441
    in_file.close()
    in_file = open(prefix + '/1/1-4-4-1_idx.pickle', 'rb')
    idx1441 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/1/1-0-2-0-1.adjlist', 'r')
    adjlist10201 = [line.strip() for line in in_file]
    adjlist10201 = adjlist10201
    in_file.close()
    in_file = open(prefix + '/1/1-0-2-0-1_idx.pickle', 'rb')
    idx10201 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/1/1-0-3-0-1.adjlist', 'r')
    adjlist10301 = [line.strip() for line in in_file]
    adjlist10301 = adjlist10301
    in_file.close()
    in_file = open(prefix + '/1/1-0-3-0-1_idx.pickle', 'rb')
    idx10301 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/1/1-0-4-0-1.adjlist', 'r')
    adjlist10401 = [line.strip() for line in in_file]
    adjlist10401 = adjlist10401
    in_file.close()
    in_file = open(prefix + '/1/1-0-4-0-1_idx.pickle', 'rb')
    idx10401 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/1/1-4-0-4-1.adjlist', 'r')
    adjlist14041 = [line.strip() for line in in_file]
    adjlist14041 = adjlist14041
    in_file.close()
    in_file = open(prefix + '/1/1-4-0-4-1_idx.pickle', 'rb')
    idx14041 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/1/1-4-2-4-1.adjlist', 'r')
    adjlist14241 = [line.strip() for line in in_file]
    adjlist14241 = adjlist14241
    in_file.close()
    in_file = open(prefix + '/1/1-4-2-4-1_idx.pickle', 'rb')
    idx14241 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/1/1-4-3-4-1.adjlist', 'r')
    adjlist14341 = [line.strip() for line in in_file]
    adjlist14341 = adjlist14341
    in_file.close()
    in_file = open(prefix + '/1/1-4-3-4-1_idx.pickle', 'rb')
    idx14341 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/1/1-0-2-2-0-1.adjlist', 'r')
    adjlist102201 = [line.strip() for line in in_file]
    adjlist102201 = adjlist102201
    in_file.close()
    in_file = open(prefix + '/1/1-0-2-2-0-1_idx.pickle', 'rb')
    idx102201 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/1/1-0-3-3-0-1.adjlist', 'r')
    adjlist103301 = [line.strip() for line in in_file]
    adjlist103301 = adjlist103301
    in_file.close()
    in_file = open(prefix + '/1/1-0-3-3-0-1_idx.pickle', 'rb')
    idx103301 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/1/1-0-4-4-0-1.adjlist', 'r')
    adjlist104401 = [line.strip() for line in in_file]
    adjlist104401 = adjlist104401
    in_file.close()
    in_file = open(prefix + '/1/1-0-4-4-0-1_idx.pickle', 'rb')
    idx104401 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/1/1-4-0-0-4-1.adjlist', 'r')
    adjlist140041 = [line.strip() for line in in_file]
    adjlist140041 = adjlist140041
    in_file.close()
    in_file = open(prefix + '/1/1-4-0-0-4-1_idx.pickle', 'rb')
    idx140041 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/1/1-4-2-2-4-1.adjlist', 'r')
    adjlist142241 = [line.strip() for line in in_file]
    adjlist142241 = adjlist142241
    in_file.close()
    in_file = open(prefix + '/1/1-4-2-2-4-1_idx.pickle', 'rb')
    idx142241 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/1/1-4-3-3-4-1.adjlist', 'r')
    adjlist143341 = [line.strip() for line in in_file]
    adjlist143341 = adjlist143341
    in_file.close()
    in_file = open(prefix + '/1/1-4-3-3-4-1_idx.pickle', 'rb')
    idx143341 = pickle.load(in_file)
    in_file.close()

    # ======================================================

    in_file = open(prefix + '/2/2-2.adjlist', 'r')
    adjlist22 = [line.strip() for line in in_file]
    adjlist22 = adjlist22
    in_file.close()
    in_file = open(prefix + '/2/2-2_idx.pickle', 'rb')
    idx22 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/2/2-0-2.adjlist', 'r')
    adjlist202 = [line.strip() for line in in_file]
    adjlist202 = adjlist202
    in_file.close()
    in_file = open(prefix + '/2/2-0-2_idx.pickle', 'rb')
    idx202 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/2/2-4-2.adjlist', 'r')
    adjlist242 = [line.strip() for line in in_file]
    adjlist242 = adjlist242
    in_file.close()
    in_file = open(prefix + '/2/2-4-2_idx.pickle', 'rb')
    idx242 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/2/2-0-0-2.adjlist', 'r')
    adjlist2002 = [line.strip() for line in in_file]
    adjlist2002 = adjlist2002
    in_file.close()
    in_file = open(prefix + '/2/2-0-0-2_idx.pickle', 'rb')
    idx2002 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/2/2-4-4-2.adjlist', 'r')
    adjlist2442 = [line.strip() for line in in_file]
    adjlist2442 = adjlist2442
    in_file.close()
    in_file = open(prefix + '/2/2-4-4-2_idx.pickle', 'rb')
    idx2442 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/2/2-0-1-0-2.adjlist', 'r')
    adjlist20102 = [line.strip() for line in in_file]
    adjlist20102 = adjlist20102
    in_file.close()
    in_file = open(prefix + '/2/2-0-1-0-2_idx.pickle', 'rb')
    idx20102 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/2/2-0-3-0-2.adjlist', 'r')
    adjlist20302 = [line.strip() for line in in_file]
    adjlist20302 = adjlist20302
    in_file.close()
    in_file = open(prefix + '/2/2-0-3-0-2_idx.pickle', 'rb')
    idx20302 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/2/2-0-4-0-2.adjlist', 'r')
    adjlist20402 = [line.strip() for line in in_file]
    adjlist20402 = adjlist20402
    in_file.close()
    in_file = open(prefix + '/2/2-0-4-0-2_idx.pickle', 'rb')
    idx20402 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/2/2-4-0-4-2.adjlist', 'r')
    adjlist24042 = [line.strip() for line in in_file]
    adjlist24042 = adjlist24042
    in_file.close()
    in_file = open(prefix + '/2/2-4-0-4-2_idx.pickle', 'rb')
    idx24042 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/2/2-4-1-4-2.adjlist', 'r')
    adjlist24142 = [line.strip() for line in in_file]
    adjlist24142 = adjlist24142
    in_file.close()
    in_file = open(prefix + '/2/2-4-1-4-2_idx.pickle', 'rb')
    idx24142 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/2/2-4-3-4-2.adjlist', 'r')
    adjlist24342 = [line.strip() for line in in_file]
    adjlist24342 = adjlist24342
    in_file.close()
    in_file = open(prefix + '/2/2-4-3-4-2_idx.pickle', 'rb')
    idx24342 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/2/2-0-1-1-0-2.adjlist', 'r')
    adjlist201102 = [line.strip() for line in in_file]
    adjlist201102 = adjlist201102
    in_file.close()
    in_file = open(prefix + '/2/2-0-1-1-0-2_idx.pickle', 'rb')
    idx201102 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/2/2-0-3-3-0-2.adjlist', 'r')
    adjlist203302 = [line.strip() for line in in_file]
    adjlist203302 = adjlist203302
    in_file.close()
    in_file = open(prefix + '/2/2-0-3-3-0-2_idx.pickle', 'rb')
    idx203302 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/2/2-0-4-4-0-2.adjlist', 'r')
    adjlist204402 = [line.strip() for line in in_file]
    adjlist204402 = adjlist204402
    in_file.close()
    in_file = open(prefix + '/2/2-0-4-4-0-2_idx.pickle', 'rb')
    idx204402 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/2/2-4-0-0-4-2.adjlist', 'r')
    adjlist240042 = [line.strip() for line in in_file]
    adjlist240042 = adjlist240042
    in_file.close()
    in_file = open(prefix + '/2/2-4-0-0-4-2_idx.pickle', 'rb')
    idx240042 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/2/2-4-1-1-4-2.adjlist', 'r')
    adjlist241142 = [line.strip() for line in in_file]
    adjlist241142 = adjlist241142
    in_file.close()
    in_file = open(prefix + '/2/2-4-1-1-4-2_idx.pickle', 'rb')
    idx241142 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/2/2-4-3-3-4-2.adjlist', 'r')
    adjlist243342 = [line.strip() for line in in_file]
    adjlist243342 = adjlist243342
    in_file.close()
    in_file = open(prefix + '/2/2-4-3-3-4-2_idx.pickle', 'rb')
    idx243342 = pickle.load(in_file)
    in_file.close()

    # ========================================================

    in_file = open(prefix + '/3/3-3.adjlist', 'r')
    adjlist33 = [line.strip() for line in in_file]
    adjlist33 = adjlist33
    in_file.close()
    in_file = open(prefix + '/3/3-3_idx.pickle', 'rb')
    idx33 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/3/3-0-3.adjlist', 'r')
    adjlist303 = [line.strip() for line in in_file]
    adjlist303 = adjlist303
    in_file.close()
    in_file = open(prefix + '/3/3-0-3_idx.pickle', 'rb')
    idx303 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/3/3-4-3.adjlist', 'r')
    adjlist343 = [line.strip() for line in in_file]
    adjlist343 = adjlist343
    in_file.close()
    in_file = open(prefix + '/3/3-4-3_idx.pickle', 'rb')
    idx343 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/3/3-0-0-3.adjlist', 'r')
    adjlist3003 = [line.strip() for line in in_file]
    adjlist3003 = adjlist3003
    in_file.close()
    in_file = open(prefix + '/3/3-0-0-3_idx.pickle', 'rb')
    idx3003 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/3/3-4-4-3.adjlist', 'r')
    adjlist3443 = [line.strip() for line in in_file]
    adjlist3443 = adjlist3443
    in_file.close()
    in_file = open(prefix + '/3/3-4-4-3_idx.pickle', 'rb')
    idx3443 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/3/3-0-1-0-3.adjlist', 'r')
    adjlist30103 = [line.strip() for line in in_file]
    adjlist30103 = adjlist30103
    in_file.close()
    in_file = open(prefix + '/3/3-0-1-0-3_idx.pickle', 'rb')
    idx30103 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/3/3-0-2-0-3.adjlist', 'r')
    adjlist30203 = [line.strip() for line in in_file]
    adjlist30203 = adjlist30203
    in_file.close()
    in_file = open(prefix + '/3/3-0-2-0-3_idx.pickle', 'rb')
    idx30203 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/3/3-0-4-0-3.adjlist', 'r')
    adjlist30403 = [line.strip() for line in in_file]
    adjlist30403 = adjlist30403
    in_file.close()
    in_file = open(prefix + '/3/3-0-4-0-3_idx.pickle', 'rb')
    idx30403 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/3/3-4-0-4-3.adjlist', 'r')
    adjlist34043 = [line.strip() for line in in_file]
    adjlist34043 = adjlist34043
    in_file.close()
    in_file = open(prefix + '/3/3-4-0-4-3_idx.pickle', 'rb')
    idx34043 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/3/3-4-1-4-3.adjlist', 'r')
    adjlist34143 = [line.strip() for line in in_file]
    adjlist34143 = adjlist34143
    in_file.close()
    in_file = open(prefix + '/3/3-4-1-4-3_idx.pickle', 'rb')
    idx34143 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/3/3-4-2-4-3.adjlist', 'r')
    adjlist34243 = [line.strip() for line in in_file]
    adjlist34243 = adjlist34243
    in_file.close()
    in_file = open(prefix + '/3/3-4-2-4-3_idx.pickle', 'rb')
    idx34243 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/3/3-0-1-1-0-3.adjlist', 'r')
    adjlist301103 = [line.strip() for line in in_file]
    adjlist301103 = adjlist301103
    in_file.close()
    in_file = open(prefix + '/3/3-0-1-1-0-3_idx.pickle', 'rb')
    idx301103 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/3/3-0-2-2-0-3.adjlist', 'r')
    adjlist302203 = [line.strip() for line in in_file]
    adjlist302203 = adjlist302203
    in_file.close()
    in_file = open(prefix + '/3/3-0-2-2-0-3_idx.pickle', 'rb')
    idx302203 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/3/3-0-4-4-0-3.adjlist', 'r')
    adjlist304403 = [line.strip() for line in in_file]
    adjlist304403 = adjlist304403
    in_file.close()
    in_file = open(prefix + '/3/3-0-4-4-0-3_idx.pickle', 'rb')
    idx304403 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/3/3-4-0-0-4-3.adjlist', 'r')
    adjlist340043 = [line.strip() for line in in_file]
    adjlist340043 = adjlist340043
    in_file.close()
    in_file = open(prefix + '/3/3-4-0-0-4-3_idx.pickle', 'rb')
    idx340043 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/3/3-4-1-1-4-3.adjlist', 'r')
    adjlist341143 = [line.strip() for line in in_file]
    adjlist341143 = adjlist341143
    in_file.close()
    in_file = open(prefix + '/3/3-4-1-1-4-3_idx.pickle', 'rb')
    idx341143 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/3/3-4-2-2-4-3.adjlist', 'r')
    adjlist342243 = [line.strip() for line in in_file]
    adjlist342243 = adjlist342243
    in_file.close()
    in_file = open(prefix + '/3/3-4-2-2-4-3_idx.pickle', 'rb')
    idx342243 = pickle.load(in_file)
    in_file.close()

    # =====================================================

    in_file = open(prefix + '/4/4-4.adjlist', 'r')
    adjlist44 = [line.strip() for line in in_file]
    adjlist44 = adjlist44
    in_file.close()
    in_file = open(prefix + '/4/4-4_idx.pickle', 'rb')
    idx44 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/4/4-0-4.adjlist', 'r')
    adjlist404 = [line.strip() for line in in_file]
    adjlist404 = adjlist404
    in_file.close()
    in_file = open(prefix + '/4/4-0-4_idx.pickle', 'rb')
    idx404 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/4/4-1-4.adjlist', 'r')
    adjlist414 = [line.strip() for line in in_file]
    adjlist414 = adjlist414
    in_file.close()
    in_file = open(prefix + '/4/4-1-4_idx.pickle', 'rb')
    idx414 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/4/4-2-4.adjlist', 'r')
    adjlist424 = [line.strip() for line in in_file]
    adjlist424 = adjlist424
    in_file.close()
    in_file = open(prefix + '/4/4-2-4_idx.pickle', 'rb')
    idx424 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/4/4-3-4.adjlist', 'r')
    adjlist434 = [line.strip() for line in in_file]
    adjlist434 = adjlist434
    in_file.close()
    in_file = open(prefix + '/4/4-3-4_idx.pickle', 'rb')
    idx434 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/4/4-0-0-4.adjlist', 'r')
    adjlist4004 = [line.strip() for line in in_file]
    adjlist4004 = adjlist4004
    in_file.close()
    in_file = open(prefix + '/4/4-0-0-4_idx.pickle', 'rb')
    idx4004 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/4/4-1-1-4.adjlist', 'r')
    adjlist4114 = [line.strip() for line in in_file]
    adjlist4114 = adjlist4114
    in_file.close()
    in_file = open(prefix + '/4/4-1-1-4_idx.pickle', 'rb')
    idx4114 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/4/4-2-2-4.adjlist', 'r')
    adjlist4224 = [line.strip() for line in in_file]
    adjlist4224 = adjlist4224
    in_file.close()
    in_file = open(prefix + '/4/4-2-2-4_idx.pickle', 'rb')
    idx4224 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/4/4-3-3-4.adjlist', 'r')
    adjlist4334 = [line.strip() for line in in_file]
    adjlist4334 = adjlist4334
    in_file.close()
    in_file = open(prefix + '/4/4-3-3-4_idx.pickle', 'rb')
    idx4334 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/4/4-0-1-0-4.adjlist', 'r')
    adjlist40104 = [line.strip() for line in in_file]
    adjlist40104 = adjlist40104
    in_file.close()
    in_file = open(prefix + '/4/4-0-1-0-4_idx.pickle', 'rb')
    idx40104 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/4/4-0-2-0-4.adjlist', 'r')
    adjlist40204 = [line.strip() for line in in_file]
    adjlist40204 = adjlist40204
    in_file.close()
    in_file = open(prefix + '/4/4-0-2-0-4_idx.pickle', 'rb')
    idx40204 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/4/4-0-3-0-4.adjlist', 'r')
    adjlist40304 = [line.strip() for line in in_file]
    adjlist40304 = adjlist40304
    in_file.close()
    in_file = open(prefix + '/4/4-0-3-0-4_idx.pickle', 'rb')
    idx40304 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/4/4-1-0-1-4.adjlist', 'r')
    adjlist41014 = [line.strip() for line in in_file]
    adjlist41014 = adjlist41014
    in_file.close()
    in_file = open(prefix + '/4/4-1-0-1-4_idx.pickle', 'rb')
    idx41014 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/4/4-2-0-2-4.adjlist', 'r')
    adjlist42024 = [line.strip() for line in in_file]
    adjlist42024 = adjlist42024
    in_file.close()
    in_file = open(prefix + '/4/4-2-0-2-4_idx.pickle', 'rb')
    idx42024 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/4/4-3-0-3-4.adjlist', 'r')
    adjlist43034 = [line.strip() for line in in_file]
    adjlist43034 = adjlist43034
    in_file.close()
    in_file = open(prefix + '/4/4-3-0-3-4_idx.pickle', 'rb')
    idx43034 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/4/4-0-1-1-0-4.adjlist', 'r')
    adjlist401104 = [line.strip() for line in in_file]
    adjlist401104 = adjlist401104
    in_file.close()
    in_file = open(prefix + '/4/4-0-1-1-0-4_idx.pickle', 'rb')
    idx401104 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/4/4-0-2-2-0-4.adjlist', 'r')
    adjlist402204 = [line.strip() for line in in_file]
    adjlist402204 = adjlist402204
    in_file.close()
    in_file = open(prefix + '/4/4-0-2-2-0-4_idx.pickle', 'rb')
    idx402204 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/4/4-0-3-3-0-4.adjlist', 'r')
    adjlist403304 = [line.strip() for line in in_file]
    adjlist403304 = adjlist403304
    in_file.close()
    in_file = open(prefix + '/4/4-0-3-3-0-4_idx.pickle', 'rb')
    idx403304 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/4/4-1-0-0-1-4.adjlist', 'r')
    adjlist410014 = [line.strip() for line in in_file]
    adjlist410014 = adjlist410014
    in_file.close()
    in_file = open(prefix + '/4/4-1-0-0-1-4_idx.pickle', 'rb')
    idx410014 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/4/4-2-0-0-2-4.adjlist', 'r')
    adjlist420024 = [line.strip() for line in in_file]
    adjlist420024 = adjlist420024
    in_file.close()
    in_file = open(prefix + '/4/4-2-0-0-2-4_idx.pickle', 'rb')
    idx420024 = pickle.load(in_file)
    in_file.close()

    in_file = open(prefix + '/4/4-3-0-0-3-4.adjlist', 'r')
    adjlist430034 = [line.strip() for line in in_file]
    adjlist430034 = adjlist430034
    in_file.close()
    in_file = open(prefix + '/4/4-3-0-0-3-4_idx.pickle', 'rb')
    idx430034 = pickle.load(in_file)
    in_file.close()

    # ===========================================================

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    prefixr = 'utils/new'
    dis2mi_train_val_test_pos = np.load(prefixr + '/dis2mi_train_val_test_pos.npz')
    dis2mi_train_val_test_neg = np.load(prefixr + '/dis2mi_train_val_test_neg.npz')
    dis2circ_train_val_test_neg = np.load(prefixr + '/dis2circ_train_val_test_neg.npz')
    dis2circ_train_val_test_pos = np.load(prefixr + '/dis2circ_train_val_test_pos.npz')
    dis2lnc_train_val_test_neg = np.load(prefixr + '/dis2lnc_train_val_test_neg.npz')
    dis2lnc_train_val_test_pos = np.load(prefixr + '/dis2lnc_train_val_test_pos.npz')
    dis2gene_train_val_test_neg = np.load(prefixr + '/dis2gene_train_val_test_neg.npz')
    dis2gene_train_val_test_pos = np.load(prefixr + '/dis2gene_train_val_test_pos.npz')
    mi2circ_train_val_test_neg = np.load(prefixr + '/mi2circ_train_val_test_neg.npz')
    mi2circ_train_val_test_pos = np.load(prefixr + '/mi2circ_train_val_test_pos.npz')
    mi2lnc_train_val_test_neg = np.load(prefixr + '/mi2lnc_train_val_test_neg.npz')
    mi2lnc_train_val_test_pos = np.load(prefixr + '/mi2lnc_train_val_test_pos.npz')
    mi2gene_train_val_test_neg = np.load(prefixr + '/mi2gene_train_val_test_neg.npz')
    mi2gene_train_val_test_pos = np.load(prefixr + '/mi2gene_train_val_test_pos.npz')
    print("ok")
    return [
        [
            adjlist00, adjlist010, adjlist020, adjlist030, adjlist040, adjlist0110, adjlist0220,
            adjlist0330, adjlist0440,adjlist01410,adjlist02420,adjlist03430, adjlist04140, adjlist04240, adjlist04340,
            # adjlist014410,adjlist024420,adjlist034430,adjlist041140, adjlist042240, adjlist043340
        ], [
            adjlist11, adjlist101, adjlist141, adjlist1001, adjlist1441, adjlist10201,
            adjlist10301, adjlist10401, adjlist14041, adjlist14241, adjlist14341,
            # adjlist102201, adjlist103301, adjlist104401, adjlist140041, adjlist142241, adjlist143341
        ], [
            adjlist22, adjlist202, adjlist242, adjlist2002, adjlist2442, adjlist20102,
            adjlist20302, adjlist20402, adjlist24042, adjlist24142, adjlist24342,
            # adjlist201102, adjlist203302, adjlist204402, adjlist240042, adjlist241142, adjlist243342
        ], [
            adjlist33, adjlist303, adjlist343, adjlist3003, adjlist3443, adjlist30103,
            adjlist30203, adjlist30403, adjlist34043, adjlist34143, adjlist34243,
            # adjlist301103, adjlist302203, adjlist304403, adjlist340043, adjlist341143, adjlist342243
        ], [
            adjlist44, adjlist404, adjlist414, adjlist424, adjlist434, adjlist4004, adjlist4114, adjlist4224,
            adjlist4334, adjlist40104, adjlist40204, adjlist40304,
            adjlist41014,adjlist42024, adjlist43034,
            # adjlist401104,
            # adjlist402204,adjlist403304, adjlist410014, adjlist420024, adjlist430034
        ]
        ], [
        [
            idx00, idx010, idx020, idx030, idx040, idx0110, idx0220, idx0330,
            idx0440,
            idx01410,
            idx02420, idx03430, idx04140, idx04240,
            idx04340
            # , idx014410,idx024420,idx034430, idx041140,
            # idx042240, idx043340
        ], [
            idx11, idx101, idx141, idx1001, idx1441, idx10201, idx10301,
            idx10401, idx14041, idx14241, idx14341
            # , idx102201, idx103301,
            # idx104401, idx140041, idx142241, idx143341
        ], [
            idx22, idx202, idx242, idx2002, idx2442, idx20102,
            idx20302, idx20402, idx24042, idx24142, idx24342,
            # idx201102, idx203302, idx204402, idx240042, idx241142, idx243342
        ], [
            idx33, idx303, idx343, idx3003, idx3443, idx30103,
            idx30203, idx30403, idx34043, idx34143, idx34243,
            # idx301103, idx302203, idx304403, idx340043, idx341143, idx342243
        ], [
            idx44, idx404, idx414, idx424, idx434, idx4004, idx4114, idx4224, idx4334,
            idx40104, idx40204, idx40304,
            idx41014,
            idx42024,
            idx43034
            # ,idx401104, idx402204, idx403304, idx410014,
            # idx420024,idx430034
        ]
        ],adjM, type_mask, dis2mi_train_val_test_pos, dis2mi_train_val_test_neg, dis2circ_train_val_test_neg, dis2circ_train_val_test_pos, dis2lnc_train_val_test_neg, dis2lnc_train_val_test_pos, dis2gene_train_val_test_neg, dis2gene_train_val_test_pos, mi2circ_train_val_test_neg, mi2circ_train_val_test_pos, mi2lnc_train_val_test_neg, mi2lnc_train_val_test_pos, mi2gene_train_val_test_neg, mi2gene_train_val_test_pos
