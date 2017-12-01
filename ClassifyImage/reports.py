#!/usr/local/bin/python2.7
""" Animal image classification reports engine. """
#
# Copyright 2017 Emilie Cassar. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
#
# Animal image classification Reports Engine.
#
# ==============================================================================
# Change Log:
# -----------
#   Date   Description
# -------- ---------------------------------------------------------------------
# 11/12/17 Initial Creation Date
#
# ==============================================================================
#
# pylint: disable=unused-import
# pylint: disable=invalid-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sys
from datetime import datetime
from operator import itemgetter
import ClassifyImage.classify as Classify
import ClassifyImage.testdata as TestData

try:
    from reportlab.platypus import TableStyle, Table, SimpleDocTemplate, Paragraph
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.graphics.barcode import code39, code128, code93
    from reportlab.graphics.barcode import eanbc, qr, usps
    from reportlab.graphics.shapes import Drawing
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import mm
    from reportlab.graphics import renderPDF
except ImportError:
    logtxt = ('Reportlab was not found. ' +
              'To export to pdf you need to have reportlab installed. ' +
              'Install it using pip install reportlab')
    print(logtxt)
    sys.exit(1)
try:
    from PollyReports import Report, Band, Element, Rule, SumElement
except ImportError:
    logtxt = ('PollyReports was not found. ' +
              'To export to pdf you need to have pollyreports installed. ' +
              'Install it using pip install pollyreports')
    print(logtxt)
    sys.exit(1)
try:
    from prettytable import PrettyTable
except ImportError:
    logtxt = ('PrettyTable was not found. ' +
              'To display ASCII table you need to have prettytable installed. ' +
              'Install it using pip install prettytable')
    print(logtxt)
    sys.exit(1)
try:
    from tinydb import TinyDB, Query
except ImportError:
    logtxt = ('TinyDB was not found. ' +
              'To use reports, you need to have TinyDB installed. ' +
              'Install it using pip install tinydb')
    print(logtxt)
    sys.exit(1)

try:
    import rollbar
except ImportError:
    LOG_TXT = ('Rollbar was not found. ' +
               'Rollbar provided automatic error reporting and needed. ' +
               'Install it using pip install rollbar')
    print(LOG_TXT)
    sys.exit(1)

REPORT_FILE = "sample_report.pdf"
yearset = set()
animallist = list()
ANIMALDICT = {}
reportdict = {}
reportlist = list()
monthdict = {}
month = {}
days = list()
mlist = list()
monthset = set()
dateset = set()
newlist = list()

def pagecount(obj):
    """ Print number of pages created on terminal """
    sys.stdout.write("%d.."
                     % obj.parent.report.pagenumber)
    sys.stdout.flush()

def animalpdfreport(animaldata, reportfile):
    """ Create the PDF report using ReportLab and PollyReports """
    rpt = Report(animaldata)
    rpt.detailband = Band([
        #Element((36, 0), ("Helvetica", 10), key = "date"),
        Element((85, 0), ("Helvetica", 10),
                key="animal", align="right", getvalue=lambda x: x["animal"].title()),
        Element((235, 0), ("Helvetica", 10),
                key="count", align="right"),
    ])
    rpt.pageheader = Band([
        Element((36, 0), ("Times-Bold", 16),
                text="GLSR Animal Classification",
                onrender=pagecount),
        Element((50, 24), ("Helvetica-Bold", 11),
                text="Animal"),
        Element((250, 24), ("Helvetica-Bold", 11),
                text="Count", align="right"),
        Rule((36, 42), 6.5*72, thickness=2),
    ])
    rpt.pagefooter = Band([
        Element((72*6.5, 0), ("Times-Bold", 16),
                text="BSA - Capitol Area Council", align="right"),
        Element((36, 16), ("Helvetica-Bold", 11),
                sysvar="pagenumber",
                format=lambda x: "Page %d" % x),
    ])
    rpt.reportfooter = Band([
        Rule((36, 4), 214),
        Element((36, 4), ("Helvetica-Bold", 12),
                text="Grand Total"),
        SumElement((235, 4), ("Helvetica-Bold", 12),
                   key="count", align="right"),
        Element((36, 16), ("Helvetica-Bold", 12),
                text=""),
    ])
    rpt.groupheaders = [
        Band([
            Rule((36, 20), 6.5*72),
            Element((36, 4), ("Helvetica-Bold", 11),
                    getvalue=lambda x: x["date"],
                    format=lambda x:
                    "Image capture date: %s" % (
                    	   datetime.strptime(x, '%Y/%m/%d').strftime("%A %B %d, %Y"))),
        ], getvalue=lambda x: x["date"]),
    ]
    rpt.groupfooters = [
        Band([
            Rule((36, 4), 214),
            Element((36, 4), ("Helvetica-Bold", 10),
                    getvalue=lambda x: x["date"],
                    format=lambda x: "Subtotal"),
            SumElement((235, 4), ("Helvetica-Bold", 10),
                       key="count", align="right"),
            Element((36, 16), ("Helvetica-Bold", 10),
                    text=""),
        ],
             getvalue=lambda x: x["date"],
             newpageafter=0),
    ]

    sys.stdout.write("Report Starting...\nGenerating page ")
    canvas = Canvas(reportfile, (72*8.5, 72*11))
    canvas = create_qr_code(canvas)
    rpt.generate(canvas)
    canvas.save()
    print("\nDone.")

def uniquedate(valdate):
    """ Find unique years in DB """
    dateyear, _, _ = valdate.split(':')
    yearset.add(dateyear)
    return dateyear

def uniquemonth(valdate, yearmonth):
    """ Find unique months for a unique year in DB """
    dateyear, datemonth, _ = valdate.split(':')
    if dateyear == yearmonth:
        monthset.add(datemonth)
    return datemonth

def uniquemonthdate(valdate, yearmonthdate, monthdate):
    """ Find unique days for a unique month, unique year in DB """
    dateyear, datemonth, dateday = valdate.split(':')
    if (dateyear == yearmonthdate) & (datemonth == monthdate):
        dateset.add(dateday)
    return dateday

def uniqueanimal(valanimal, yearanimal, monthanimal, day):
    """ Find animal for this year, month, day """
    global ANIMALDICT
    try:
        animalcount = ANIMALDICT[yearanimal][monthanimal][day][valanimal]
        animalcount += 1
        ANIMALDICT[yearanimal][monthanimal][day][valanimal] = animalcount
    except KeyError:
    	ANIMALDICT[yearanimal][monthanimal][day][valanimal] = 1

def create_qr_code(canvas):
    """ draw a QR code """
    qr_code = qr.QrCodeWidget(Classify.LEGAL_WS)
    bounds = qr_code.getBounds()
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    qrdrawing = Drawing(45, 45, transform=[45./width, 0, 0, 45./height, 0, 0])
    qrdrawing.add(qr_code)
    renderPDF.draw(qrdrawing, canvas, 15, 405)
    return canvas

def run_report(flagtext, flagpdf, testdata):
    """ Run and creates the reports """
    global ANIMALDICT
    print("classify_image Reports Version " + Classify.VERSION + " running...")
    if not testdata:
        animaldb = TinyDB('hornaday_db.json', default_table='imagetable')
        animaltable = animaldb.table('animaltable')
        if len(animaltable) == 0:
        	print("Error: Classification database is empty. Classify some images or use the -wtd command line flag for using test data instead.")
        	return
        year = Query()
        result = animaltable.search(year.date.test(uniquedate))
        yearlist = list(sorted(yearset))
        # Loop trough the years
        for year_in_list in yearlist:
            result = animaltable.search(year.date.test(uniquemonth, year_in_list))
            mlist = list(sorted(monthset))
            month = {mlist[i]: days for i in range(0, len(mlist), 1)}
            monthdict[year_in_list] = month
        # Loop trough years and months
        for month_in_dict in monthdict:
            monthlist = monthdict[month_in_dict]
            for month_in_list in monthlist:
                result = animaltable.search(
                    year.date.test(uniquemonthdate, month_in_dict, month_in_list))
                dlist = list(sorted(dateset))
                monthdict[month_in_dict][month_in_list] = dlist
        # Sort this dict using an ordereddict
        omd = collections.OrderedDict(sorted(monthdict.items()))
        # Now that we have years, months and days, calculate the total for animals
        for yil in omd:
            mlist = omd[yil]
            if not yil in ANIMALDICT:
                ANIMALDICT[yil] = {}
            for mil in mlist:
                dlist = omd[yil][mil]
                if not mil in ANIMALDICT[yil]:
                    ANIMALDICT[yil][mil] = {}
                for dil in dlist:
                    if not dil in ANIMALDICT[yil][mil]:
                        ANIMALDICT[yil][mil][dil] = {}
                    result = animaltable.search(
                        (year.date == yil +
                         ":" + mil + ":" + dil) &
                        (year.animal.test(uniqueanimal, yil, mil, dil)))
    else:
    	ANIMALDICT = TestData.get_animal_dict()
    	newlist = TestData.get_new_list()
    animald = collections.OrderedDict(sorted(ANIMALDICT.items()))
    ptable = PrettyTable(["Date", "Animal", "Count"])
    ptable.padding_width = 1
    ptable.align['Animal'] = "l"
    ptable.align['Count'] = "r"
    ptable.sortby = "Date"
    for yil in animald:
        mlist = animald[yil]
        for mil in mlist:
            dlist = animald[yil][mil]
            for dil in dlist:
                dbdate = yil + "/" + mil + "/" + dil
                for ail in animald[yil][mil][dil]:
                    date_object = datetime.strptime(dbdate, '%Y/%m/%d')
                    strdate = date_object.strftime("%A %B %d, %Y")
                    animallist = list()
                    animallist.append(dbdate)
                    animallist.append(ail)
                    animallist.append(animald[yil][mil][dil][ail])
                    ptable.add_row(animallist)
                    reportdict = {}
                    reportdict['date'] = dbdate
                    reportdict['animal'] = ail
                    reportdict['count'] = animald[yil][mil][dil][ail]
                    reportlist.append(reportdict)
    # Sort the report by date value
    newlist = sorted(reportlist, key=itemgetter('date'))
    if flagtext:
    	""" Text report """
        print(ptable)
    if flagpdf:
    	""" PDF report """
        animalpdfreport(newlist, REPORT_FILE)

# Main Program
if __name__ == "__main__":
    try:
        run_report(True, True, True)
    except:
        # catch-all
        rollbar.report_exc_info()
