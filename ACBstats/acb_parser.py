#!/usr/bin/python
# -*- encoding: utf-8 -*-

import urllib2
import bs4
import json

###############################################################################

def leer_webpage(url):

   #Descargamos la web
   website = urllib2.urlopen(url)
   website_html = website.read()

   tabla = [[(cell.text).encode('utf-8') for cell in row("td")]
                 for row in bs4.BeautifulSoup(website_html)("tr")]
               


   #Intemos hacer algo util con la tabla
   estadisticas=tabla[0][2].split('\n')

   partido=estadisticas[6].split('|')

   return partido

#########################################################################


print(leer_webpage("http://www.acb.com/fichas/LACB60040.php"))


