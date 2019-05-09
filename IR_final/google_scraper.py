#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# adapted from http://stackoverflow.com/questions/20716842/python-download-images-from-google-image-search
# modified based on https://gist.github.com/genekogan/ebd77196e4bf0705db51f86431099e57

from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

import re, os, argparse, sys, json, random, logging, time
logging.basicConfig(level = logging.INFO, format = '%(asctime)s : %(levelname)s : %(module)s : %(funcName)s : %(message)s')

def scraper(query, max_images, path_prefix, s_index):	

    sub_dir = str(s_index)+"_"+"_".join(query.strip().split())
    ensure_folder_exists(path_prefix)
    ensure_folder_exists(os.path.join(path_prefix,sub_dir))
    soup = get_img_search_results(query = query)
    img_urls = [ (json.loads(a.text)["ou"], json.loads(a.text)["ity"]) for a in soup.find_all("div",{"class":"rg_meta"}) ]

    for i, (img_link, file_ext) in enumerate( img_urls[:max_images] ):
		
        if len(file_ext) == 0:
            file_ext = "jpg"

        logging.info("Handling %d" %(i+1))
        out_name = os.path.join(path_prefix , sub_dir, str(s_index) + "_" + str(i+1) + "." + file_ext)
        os.system("wget -q \"%s\" -O %s > log.txt" %(img_link, out_name))
        time.sleep(0.2 + random.random()/4)		
        '''
        logging.info("Handling %s" %(img_link))
        raw_img = urlopen( Request( img_link, headers = get_random_header() ) ).read()
        time.sleep(0.2 + random.random()/4)		
        #with open(os.path.join(path_prefix , "_".join(query.strip().split()) + "_" + str(i+1) + "." + file_ext), 'wb') as f:
        with open(os.path.join(path_prefix , str(s_index)+"_"+"_".join(query.strip().split()), str(s_index) + "_" + str(i+1) + "." + file_ext), 'wb') as f:
            f.write(raw_img)		
        '''

def get_img_search_results(query):
	return get_soup(url = make_search_url(query), header = get_random_header())

def get_random_header():
	return { 'User-Agent':random.choice(USER_AGENT_LIST), 'Referer':random.choice(REFERER_LIST) }

def make_search_url(query): # query: string like "fat cat"
	return "https://" + random.choice(GOOGLE_LIST) + "/search?q=" + '+'.join(query.strip().split()) + "&source=lnms&tbm=isch"

def get_soup(url, header):
	return BeautifulSoup( urlopen( Request(url, headers = header) ), 'html.parser' )

def ensure_folder_exists(folder_name):
	par = os.path.abspath(os.path.join(folder_name, os.pardir))
	
	if par != '/':
		ensure_folder_exists(par)

	if not (os.path.exists(folder_name) or os.path.isdir(folder_name)):
		logging.info("mkdir " + str(folder_name))
		os.mkdir(folder_name)

USER_AGENT_LIST = [
	"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36",
	"Mozilla/5.0 (Windows; U; Windows NT 5.1; de; rv:1.9.2.3) Gecko/20100401 Firefox/3.6.3 (.NET CLR 3.5.30729) (Prevx 3.0.5)",
	"Mozilla/5.0 (Windows; U; Windows NT 5.1; de; rv:1.9.2.3) Gecko/20100401 Firefox/3.6.3 (FM Scene 4.6.1)",
	"Mozilla/5.0 (Linux; Android 6.0.1; E6653 Build/32.2.A.0.253) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.98 Mobile Safari/537.36",
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246",
	"Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36",
	"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.111 Safari/537.36",
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9",
	"Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1",
	"Mozilla/5.0 (Linux; Android 6.0.1; SM-G920V Build/MMB29K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.98 Mobile Safari/537.36"
]

REFERER_LIST = [ 
	"http://www.python.org/",
	"https://stackoverflow.com/questions/30703485/dataset-for-emotion-detection-by-text",
	"https://stackoverflow.com/questions/28163726/anyone-knows-about-text-based-emotion-detection-systems-that-offer-a-demo?rq=1",
	"https://stackoverflow.com/questions/27943396/using-wn-affect-to-detect-emotion-mood-of-a-string",
	"https://stackoverflow.com/questions/5062032/audio-analysis-to-detect-human-voice-gender-age-and-emotion-any-prior-open",
	"http://www.crazy-photoshop.com/?p=74047",
	"http://www.networkcomputing.com/wireless-infrastructure/how-does-mu-mimo-work/748964231",
	"http://www.cc.ntu.edu.tw/chinese/epaper/0024/20130320_2409.html",
	"https://kknews.cc/tech/vmvjm32.html",
	"http://eriknaso.com/2015/10/27/you-dont-have-to-shoot-log-all-the-time-right/"
]

GOOGLE_LIST = [ 
	"www.google.co.in",
	"www.google.com",
	"www.google.com.tw",
	"www.google.co.jp",
	"www.google.ca"
]

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Scrape Google images')
	parser.add_argument('-s', '--search', default='cats', type=str, help='search term')
	parser.add_argument('-n', '--num_images', default=5, type=int, help='num images to save')
	parser.add_argument('-d', '--directory', default='image_test/', type=str, help='save directory')
	parser.add_argument('-i', '--s_index', default=0, type=int, help='sentence index')
	args = parser.parse_args()

	scraper(query = args.search.strip(), max_images = args.num_images, path_prefix = args.directory, s_index = args.s_index)

	quit()
