import os
import pdfbox

def convert_txt(indir, outdir, pdfname):
    print("\n")
    print(">>>>>>>>>>>>>>>>>>>>>>>> Installing PDFBox... <<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    os.system('pip install python-pdfbox')
    
    print("\n")
    print(">>>>>>>>>>>>>>>>>>>>>>>> Converting File... <<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("  => Inputing a document...")
    # remove single quotes from file name
    pdfname = pdfname.replace("'", "")
    #os.system('bash src/rename.sh')
    # extract text using PDFBox
    input_fp = os.path.join(indir, pdfname)
    temp_txt = input_fp.replace('.pdf', '.txt')
    temp_txt = temp_txt.replace(' ', '_')
    p = pdfbox.PDFBox()
    p.extract_text(input_fp, temp_txt)
    
    # make a directory if outdir does not exist
    command = 'mkdir -p ' + outdir
    os.system(command)
    
    print("  => Converting pdf to txt...")
    textname = pdfname.replace('.pdf', '_converted.txt')
    textname = textname.replace(' ', '_')
    output_fp = os.path.join(outdir, textname)
    output_txt = open(output_fp, 'w')
    # concatenate split lines
    with open(temp_txt, 'rb') as f:
        for line in f:
            line = line.decode()
            if len(line) >= 2 and line[-2] == '-':
                output_txt.write(line[:-2])
            else:
                output_txt.write(line[:-1] + ' ')
    output_txt.close()
    
    # save output
    command = 'rm ' + temp_txt
    os.system(command)
    print(" => Done! File is saved as '" + output_fp + "'")
    print("\n")
    return
