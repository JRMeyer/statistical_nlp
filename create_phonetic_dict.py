from tokenize_corpus import tokenize_line
import re

def save_pronunciation_dict(tokens, lookupTable, lang):
    outFile = open('language.dict', mode='wt', encoding='utf-8') 

    for token in sorted(set(tokens)):
        phonemes=token

        ###
        ##
        #
        
        if lang == 'kyrgyz':
            # plosives followed by front/back vowels
            phonemes = re.sub(r'к([аоуы])', r'kh \1', phonemes)
            phonemes = re.sub(r'к([иеэөү])', r'k \1', phonemes)
            phonemes = re.sub(r'г([аоуы])', r'gh \1', phonemes)
            phonemes = re.sub(r'г([иеэөү])', r'g \1', phonemes)

            # syllable final plosives preceded by front/back vowels
            phonemes = re.sub(r'([аоуы])к([^аоуыиеэөү])', r'\1kh \2', phonemes)
            phonemes = re.sub(r'([иеэөү])к([^аоуыиеэөү])', r'\1k \2', phonemes)
            phonemes = re.sub(r'([аоуы])г([^аоуыиеэөү])', r'\1gh \2', phonemes)
            phonemes = re.sub(r'([иеэөү])г([^аоуыиеэөү])', r'\1g \2', phonemes)

            # word final plosives preceded by front/back vowels
            phonemes = re.sub(r'([аоуы])к($)', r'\1kh', phonemes)
            phonemes = re.sub(r'([иеэөү])к($)', r'\1k', phonemes)
            phonemes = re.sub(r'([аоуы])г($)', r'\1gh', phonemes)
            phonemes = re.sub(r'([иеэөү])г($)', r'\1g', phonemes)

            # /b/ between back vowels goes to [w] 
            phonemes = re.sub(r'([аоуы])б([аоуы])', r'\1b \2', phonemes)

            # diphthongs
            phonemes = re.sub(r'ой', r'o i ', phonemes)
            phonemes = re.sub(r'ай', r'a i ', phonemes)

        elif lang == 'kazakh':
            pass

        #
        ##
        ###
        
        for character in phonemes:
            if character in lookupTable:
                phonemes = re.sub(character, lookupTable[character], phonemes)

        # in case we added a space to the end of the sequence
        phonemes = phonemes.strip()
        
        print((token + ' ' + phonemes), end='\n', file=outFile)


kyrgyz_table = {'а':'a ', # back vowels
               'о':'o ',
               'у':'u ',
               'ы':'ih ',
               'и':'i ', # front vowels
               'е':'e ',
               'э':'e ',
               'ө':'oe ',
               'ү':'y ',
               'ю':'j u ', # glide vowels
               'я':'j a ',
               'ё':'j o ',
               'п':'p ', # bilabials
               'б':'b ',
               'д':'d ', # coronals
               'т':'t ',
               'к':'k ', # velars
               'г':'g ',
               'х':'h ',
               'ш':'sh ', # (alveo)(palatals)
               'щ':'sh ',
               'ж':'zh ',
               'з':'z ', 
               'с':'s ',
               'ц':'ts ', # affricates
               'ч':'ch ',
               'й':'j ', # glides
               'л':'l ',
               'м':'m ', # nasals
               'н':'n ',
               'ң':'ng ',
               'ф':'f ', # labiodentals
               'в':'v ',
               'р':'r ', # trill
               'ъ':' ',
               'ь':' '}
        


kazakh_table = {'б' : 'b ',
                'в' : 'v ',
                'г' : 'g ',
                'д' : 'd ',
                'з' : 'z ',
                'к' : 'k ',
                'л' : 'l ',
                'м' : 'm ',
                'н' : 'n ',
                'п' : 'p ',
                'р' : 'r ',
                'с' : 's ',
                'т' : 't ',
                'ф' : 'f ',
                'х' : 'h ',
                'һ' : 'h ', 
                'ж' : 'zh ',
                'ц' : 'ts ',
                'ч' : 'ch ',
                'ш' : 'sh ',
                'щ' : 'sh ',
                'й' : 'j ',
                'ғ' : 'gh ',
                'қ' : 'kh ',
                'ң' : 'ng ',
                'а' : 'a ', # vowels
                'ә' : 'a ',
                'у' : 'u ',
                'ү' : 'y ',
                'ұ' : 'o ',
                'о' : 'o ',
                'ө' : 'oe ',
                'э' : 'e ',
                'е' : 'e ',
                'и' : 'i ',
                'і' : 'ih ',
                'ы' : 'ih ',
                'я' : 'j a ', # glide + vowels
                'ё' : 'j o ',
                'ю' : 'j u ',
                'ь' : '',
                'ъ' : ''}


        
if __name__ == '__main__':
    fileName = input('enter file path here: ')
    f = open(fileName)

    # regex pattern to match everything that isn't a letter
    pattern = re.compile('[\W_0-9]+', re.UNICODE)
    
    tokens=[]
    for line in f:
        for token in tokenize_line(line, n=1, tags=False):
            token = pattern.sub('', token)
            tokens.append(token)
    save_pronunciation_dict(tokens, lookupTable=kazakh_table, lang='kazakh')
