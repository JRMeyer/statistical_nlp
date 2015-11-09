from tokenize_corpus import tokenize_line
import re

def save_pronunciation_dict(tokens, lookupTable):
    outFile = open('kyrgyz.dict', mode='wt', encoding='utf-8') 

    for token in sorted(set(tokens)):
        phonemes=token

        # palatal/uvular plosives followed by front/back vowels
        phonemes = re.sub(r'к([аоуы])', r'kh \1', phonemes)
        phonemes = re.sub(r'к([иеэөү])', r'k \1', phonemes)
        phonemes = re.sub(r'г([аоуы])', r'gh \1', phonemes)
        phonemes = re.sub(r'г([иеэөү])', r'g \1', phonemes)

        # syllable final palatal/velar plosives preceded by front/back vowels
        phonemes = re.sub(r'([аоуы])к([^аоуыиеэөү])', r'\1kh \2', phonemes)
        phonemes = re.sub(r'([иеэөү])к([^аоуыиеэөү])', r'\1k \2', phonemes)
        phonemes = re.sub(r'([аоуы])г([^аоуыиеэөү])', r'\1gh \2', phonemes)
        phonemes = re.sub(r'([иеэөү])г([^аоуыиеэөү])', r'\1g \2', phonemes)

        # word final palatal/velar plosives preceded by front/back vowels
        phonemes = re.sub(r'([аоуы])к($)', r'\1kh', phonemes)
        phonemes = re.sub(r'([иеэөү])к($)', r'\1k', phonemes)
        phonemes = re.sub(r'([аоуы])г($)', r'\1gh', phonemes)
        phonemes = re.sub(r'([иеэөү])г($)', r'\1g', phonemes)

        # /b/ between back vowels goes to [w] 
        phonemes = re.sub(r'([аоуы])б([аоуы])', r'\1b \2', phonemes)

        # diphthongs
        phonemes = re.sub(r'ой', r'o i ', phonemes)
        phonemes = re.sub(r'ай', r'a i ', phonemes)

        for character in phonemes:
            if character in lookupTable:
                phonemes = re.sub(character, lookupTable[character], phonemes)

        # in case we added a space to the end of the sequence
        phonemes=phonemes.strip()
        
        print((token + ' ' + phonemes), end="\n", file=outFile)


# based on Arpabet https://en.wikipedia.org/wiki/Arpabet
lookupTable = {'а':'a ',
               'о':'o ',
               'у':'u ',
               'ы':'y ',
            
               'и':'i ',
               'е':'e ',
               'э':'e ',
               'ө':'o ',
               'ү':'u ',
               
               'ю':'u ',
               'я':'a ',
               'ё':'o ',
               
               'п':'p ',
               'б':'b ',
               
               'д':'d ',
               'т':'t ',

               'к':'k ',
               'г':'g ',
               
               'ш':'sh ',
               'щ':'sch ',
               'ж':'zh ',
            
               'й':'j ',
               'л':'l ',
               'м':'m ',
               'н':'n ',
               'ң':'ng ',
               
               'з':'z ',
               'с':'s ',
               'ц':'c ',
               'ч':'ch ',
               'ф':'f ',
               'в':'v ',
               'х':'h ',
               'р':'r ',
               'ъ':' ',
               'ь':' '}
        
        
if __name__ == "__main__":
    fileName = input("enter file path here: ")
    f = open(fileName)

    # regex pattern to match everything that isn't a letter
    pattern = re.compile('[\W_0-9]+', re.UNICODE)
    
    tokens=[]
    for line in f:
        for token in tokenize_line(line, n=1, tags=False):
            token = pattern.sub('', token)
            tokens.append(token)
    save_pronunciation_dict(tokens, lookupTable)
