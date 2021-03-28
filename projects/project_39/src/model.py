import os

def autophrase(data_outdir, pdfname, outdir, filename):
    # remove single quotes from file name
    textname = pdfname.replace("'", "")
    textname = textname.replace('.pdf', '_converted.txt')
    textname = textname.replace(' ', '_')
    output_fp = os.path.join(data_outdir, textname)
    
    # copy txt file to AutoPhrase/test/testdata
    command = 'cp ' + output_fp + ' AutoPhrase/test/testdata'
    os.system(command)
    command = 'mv AutoPhrase/test/testdata/' + textname + ' AutoPhrase/test/testdata/test_raw.txt'
    os.system(command)
    print("\n")
    print(">>>>>>>>>>>>>>>>>>>>>>>> Running AutoPhrase... <<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    print("  => Running AutoPhrase on the input document...")
    # write bash script to call the target test
    with open('src/run.sh', 'w') as rsh:
        rsh.write('''cd AutoPhrase \n''')
        rsh.write('''python run.py reset \n''')
        rsh.write('''python run.py test \n''')
    os.system('bash src/run.sh')
    
    # make a directory if outdir does not exist
    command = 'mkdir -p ' + outdir
    os.system(command)
    
    # save output
    print("  => Saving results...")
    output_fp = os.path.join(outdir, filename)
    os.system('cp AutoPhrase/data/out/AutoPhrase_Result/AutoPhrase.txt ' + output_fp)
    print(" => Done! AutoPhrase output is saved as '" + output_fp + "'")
    print("\n")
    return
