mkdir -p autolibrary/documents_copy
cp -r autolibrary/documents/. autolibrary/documents_copy/.
for i in autolibrary/documents_copy/* ; do mv -v "$i" "${i/\'/}" ; done
for f in autolibrary/documents_copy/* ; do mv -v "$f" "${f// /_}" ; done
