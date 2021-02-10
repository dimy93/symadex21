import re
import numpy as np
failed_im = {}
a = []
class Image:
    def __init__( self, time, adex_classes ):
        self.time = time
        self.adex_classes = adex_classes
        self.epsilon = epsilon
    def __str__( self ):
        return str( self.adex_classes ) + " in " + str( self.time ) + " sec"
    def __repr__( self ):
        return self.__str__()

class Adex:
    def __init__( self, cl, pixels, array, epsilon ):
        self.cl = cl
        self.pixels = pixels
        self.array = array
        self.epsilon = epsilon
    def __str__( self ):
        return 'pix ' + str( self.pixels ) + " vol " + str( np.sum(np.log10(self.array)) )
    def __repr__( self ):
        return self.__str__()


with open('sm_conv_exp_12_737.txt', 'r') as logs:
#with open('6x500_exp_30.txt', 'r') as logs:
    while True:
        line = logs.readline()
        if not line:
            break
        res = re.search("img (\d+) Failed", line)
        if res:
            img_num = int( res.group(1) )
            adex = {}
            bad_classes = 0
            good_classes = 0
            while True:
                line = logs.readline()
                if not line:
                    break
                res = re.search("(\d+).(\d+) seconds", line)
                if res:
                    time = int( res.group(1) )
                    break;
                res = re.search("PGD bad attacks (48|49|50)/50", line)
                if res:
                    bad_classes += 1
                
                epsilon = None
                res = re.search("Shrank with epsilon 0.(\d+)", line)
                if res:
                    epsilon = float( '0.' + res.group(1) )
                    line = logs.readline()

                res = re.search("Adex for class (\d+)", line)
                if res:
                    adex_class = int( res.group(1) )
                    good_classes += 1
                    line = logs.readline()
                    res = re.search("# Changed pixels (\d+)", line)
                    pixels_changed = int( res.group(1) )
                    pixel_changes = ""
                    while True:
                        c = logs.read(1)
                        pixel_changes += c
                        if c == ']':
                            break
                    pixel_changes = pixel_changes[ 1 : -1 ].strip()
                    if len( pixel_changes ) == 0:
                        pixel_changes = []
                    else:
                        pixel_changes = re.split('\s+', pixel_changes)
                        pixel_changes = [ int(val) for val in pixel_changes ]
                    pixel_changes = np.array( pixel_changes, dtype=np.float64 )
                    adex[ adex_class ] = Adex( adex_class, pixels_changed, pixel_changes, epsilon )


            failed_im[img_num] = Image( time, adex )
            if not bad_classes + good_classes == 9:
                a.append( img_num )
        #if re.search("Adex for class \d+", line):
        #    a += 1
print( failed_im )
i = 0
j = 0 
avg_vol = 0.0
avg_pix = 0.0 
eps = 0.0
avg_eps = 0.0
avg_time_for_succ = 0.0
for key, val in failed_im.items():

    if len( val.adex_classes.keys() ) > 0:
        i += 1
        avg_time_for_succ += val.time
        for cl,adex in val.adex_classes.items():
            j += 1
            avg_vol += np.sum(np.log10(adex.array))
            avg_pix += adex.pixels
            if adex.epsilon:
                eps += 1
                avg_eps += adex.epsilon
avg_vol /= j
avg_pix /= j
avg_time_for_succ /= i
avg_eps /= eps
eps /= j

print( 'Adex regions', j )
print( 'Successful imgs', i )
print( 'Avg vol', avg_vol )
print( 'Avg pix', avg_pix )
print( 'Avg classes', j/i )
print( 'Portion of shrinkage', eps )
print( 'Avg eps', avg_eps )
print( 'Avg time per adex img', avg_time_for_succ )
print( 'Bad shrinking: ', a )

