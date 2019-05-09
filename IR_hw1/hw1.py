import numpy as np
import pickle
import xml.etree.ElementTree
import sys

rel_open = sys.argv[1]
query_file = sys.argv[2]
rank_list = sys.argv[3]
model_dir = sys.argv[4]
NTCIR_dir = sys.argv[5]

#file_position = "/tmp2/b02902030/IR_hw1/data/"
#model = file_position+"model/"
#query_position = file_position+"queries/"

def read_model():
    vocab_inverse = {}
    with open(model_dir+"/vocab.all") as fp:
        count = 0
        for f in fp:
            f = f.strip()
            vocab_inverse[f] = count
            count+=1

    vocab = {}
    with open(model_dir+"/vocab.all") as fp:
        count = 0
        for f in fp:
            f = f.strip()
            vocab[count] = f
            count+=1

    file_list = {}
    with open(model_dir+"/file-list") as fp:
        count = 0
        for f in fp:
            f = f.strip()
            file_list[count] = f[8:]
            count+=1

    collection = {}
    df = {}
    #with open(model+"try") as fp:
    with open(model_dir+"/inverted-file") as fp:
        p = (-1,-1)
        for f in fp:
            f = f.strip()
            a = list(map(int,f.split()))
            if len(a)==3:
                p = (a[0], a[1])
                df[p] = a[2]
            elif len(a)==2:
                if a[0] not in collection:
                    collection[a[0]] = {}
                collection[a[0]][p] = a[1]
    
    return vocab, vocab_inverse, file_list, collection, df
    '''
    with open(file_position+"model_files.pickle", "wb") as fp:
        pickle.dump(vocab, fp, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(vocab_inverse, fp, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(file_list, fp, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(collection, fp, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(df, fp, protocol=pickle.HIGHEST_PROTOCOL)
    '''

def read_query(query_file):
    e = xml.etree.ElementTree.parse(query_file).getroot()
    query_number = len(e.findall(".//concepts"))
    queries = {}
    for j in range(query_number):
        d = {}
        k = e.findall(".//concepts")[j].text.strip()[:-1].split("、")
        for kk in k:
            for kkk in kk:
                if kkk not in d:
                    d[kkk] = 0
                d[kkk]+=1
        for kk in k:
            for i in range(len(kk)-1):
                if kk[i:i+2] not in d:
                    d[kk[i:i+2]] = 0
                d[kk[i:i+2]]+=1
        
        queries[j] = d
    return queries

def cos_sim(a,b):
    if np.sum(a**2)==0 or np.sum(b**2)==0:
        return 0
    return np.dot(a,b) / np.sqrt(np.sum(a**2)*np.sum(b**2))

def main():
    vocab, vocab_inverse, file_list, collection, df = read_model()
    queries = read_query(query_file)
    
    '''
    with open(file_position+"model_files.pickle", "rb") as fp:
        vocab = pickle.load(fp)
        vocab_inverse = pickle.load(fp)
        file_list = pickle.load(fp)
        collection = pickle.load(fp)
        df = pickle.load(fp)
    '''

    #count doclen/avgdoclen for each doc
    doclen = np.zeros(len(file_list))
    for i in range(len(file_list)):
        e = xml.etree.ElementTree.parse(NTCIR_dir+"/"+file_list[i]).getroot()
        dl = 0
        for k in e.findall(".//p"):
            dl += len(k.text)
        doclen[i] = dl
    doclen = doclen/np.mean(doclen)

    ok = 3.0
    ob = 1.0

    #deal with each query
    with open(rank_list, "w") as fw:
        fw.write("query_id,retrieved_docs\n")
        for qq in range(len(queries)):
            print(qq+1)
            q = queries[qq]
            query = [] # ex: [(1,-1),(1,123),(2,246)]
            query_count = []
            for k in q.keys():
                if len(k)==1:
                    if k in vocab_inverse:
                        query.append((vocab_inverse[k],-1))
                        query_count.append(q[k])
                elif len(k)==2:
                    if k[0] in vocab_inverse and k[1] in vocab_inverse:
                        query.append((vocab_inverse[k[0]],vocab_inverse[k[1]]))
                        query_count.append(q[k])
            query_count = np.array(query_count)
            query_TFIDF = (ok+1)*query_count / (query_count+ok)
            for i in range(len(query_count)):
                if query[i] in df:
                    query_TFIDF[i] *= np.log(len(doclen)/df[query[i]])

            all_TFIDF = np.zeros((len(file_list), len(query_count)))
            cos_result = np.zeros(len(file_list))
            for f in range(len(file_list)):
                if f not in collection:
                    cos_result[f] = 0
                    all_TFIDF[f] = np.zeros(len(query_count))
                    continue
                doc_TFIDF = np.zeros(len(query_count))
                for i in range(len(query_count)):
                    if query[i] not in collection[f]:
                        doc_TFIDF[i] = 0
                    else:
                        tmp = collection[f][query[i]]
                        doc_TFIDF[i] = (ok+1)*tmp / (tmp+ok*(1-ob+ob*doclen[f]))
                        doc_TFIDF[i] *= np.log(len(doclen)/df[query[i]])
                all_TFIDF[f] = np.copy(doc_TFIDF)
                cos_result[f] = cos_sim(query_TFIDF, doc_TFIDF)

            result = np.argsort(cos_result)[::-1]

            #Rocchio feedback
            R_para = [0,0,0,0,0]
            if rel_open=="rel_on":
                R_para = [0.78, 0.2, 0.02, 10, 10000]
            else:
                R_para = [1, 0, 0, 10, 10000]
           
            #print(alpha, beta, gamma, rel_num, non_rel_num)
            new_query = np.zeros(len(query_TFIDF))
            new_query += R_para[0]*query_TFIDF
            new_query += R_para[1]*np.mean(all_TFIDF[result[:R_para[3]]], axis=0)
            new_query -= R_para[2]*np.mean(all_TFIDF[result[R_para[4]:R_para[4]+1000]], axis=0) 
            cos_result = np.zeros(len(file_list))
            for f in range(len(file_list)):
                cos_result[f] = cos_sim(new_query, all_TFIDF[f])
            result = np.argsort(cos_result)[::-1]
            
            #write in output.csv
            e = xml.etree.ElementTree.parse(query_file).getroot()
            q_digit = e.findall(".//number")[qq].text[-3:]
            fw.write(q_digit+",")
            for i in range(100):
                fw.write(file_list[result[i]][-15:].lower()+" ")
            fw.write("\n")


if __name__== '__main__':
    main()
