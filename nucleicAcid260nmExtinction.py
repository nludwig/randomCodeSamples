import sys

#singles & doubles data taken from webpage
#http://www.owczarzy.net/extinctionDNA.htm
def computeNA260nmExtinction(seq, naType):
    singles = {'A': 15400,
               'C': 7400,
               'G': 11500}
    doubles = {'AA': 27400,
               'AC': 21200,
               'AG': 25000,
               'CA': 21200,
               'CC': 14600,
               'CG': 18000,
               'GA': 25200,
               'GC': 17600,
               'GG': 21600}
    if naType == 'DNA':
        singles['T'] = 8700
        doubles['AT'] = 22800
        doubles['CT'] = 15200
        doubles['GT'] = 20000
        doubles['TA'] = 23400
        doubles['TC'] = 16200
        doubles['TG'] = 19000
        doubles['TT'] = 16800
    elif naType == 'RNA':
        singles['C'] = 7200
        singles['U'] = 9900
        doubles['AC'] = 21000
        doubles['AU'] = 24000
        doubles['CA'] = 21000
        doubles['CC'] = 14200
        doubles['CG'] = 17800
        doubles['CU'] = 16200
        doubles['GC'] = 17400
        doubles['GU'] = 21200
        doubles['UA'] = 24600
        doubles['UC'] = 17200
        doubles['UG'] = 20000
        doubles['UU'] = 19600
    else:
        exit(1)
    l = len(seq)
    singlesSum = -singles[seq[0]]
    doublesSum = 0.
    for i in range(l-1):
        singlesSum += singles[seq[i]]
        doublesSum += doubles[seq[i:i+2]]
    return doublesSum - singlesSum

def main():
    nargs = 1
    if len(sys.argv)-1 != nargs:
        print('usage: python3 nucleicAcid260nmExtinction.py SEQUENCE', file=sys.stderr)
        exit(1)
    args = sys.argv[1:]
    seq = args[0].upper()
    for c in seq:
        if c == 'T':
            naType = 'DNA'
            break
        if c == 'U':
            naType = 'RNA'
    extinction = computeNA260nmExtinction(seq, naType)
    print('extinction for seq\n{}\n{} L / (mol cm)'.format(seq.upper(), extinction))

if __name__ == '__main__':
    main()
