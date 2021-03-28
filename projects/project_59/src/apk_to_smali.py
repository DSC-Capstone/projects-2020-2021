import argparse
import os

def main():
    '''
    Usage: find "apk_folder" -type f -name '*.apk' -print0 | xargs -0 -n1 -P8 python src/apk_to_smali.py --outfolder "outfolder"
    '''
    parser = argparse.ArgumentParser(description='APK to Smali Conversion Script')
    parser.add_argument('apk_path', help='path to an apk')
    parser.add_argument('--outfolder', type=str,
                        help='Folder to unload apps, will place each app into their own folders.')
    args = parser.parse_args()
    app_name = os.path.split(args.apk_path.replace('.apk', ''))[1]
    os.system(f"apktool d -r -f --no-assets {args.apk_path} -o {os.path.join(args.outfolder, app_name)}")
#     os.system()

if __name__ == '__main__':
    main()