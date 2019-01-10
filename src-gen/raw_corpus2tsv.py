import re
import os
import sys
import time
from lxml import etree
from unidecode import unidecode


def clean(value):
    if value is None:
        return ""
    else:
        return re.sub(r"( +|\n|\t|--)", " ", unidecode(value).strip())


def xml2tsv(filepath):
    """ 
    extracts fields from Hansard XML files and outputs TSV 
    file with one row per speech act 
    """
    printable = frozenset(("i", "b"))

    dt = time.strftime("%Y%m%d")
    sys.stdout = open('../data/membercontributions-' + dt + '.tsv', 'w')
    sys.stdout.write("ID\tDATE\tBILL\tMEMBER\tCONSTITUENCY\tSPEECH_ACT") 
    sys.stdout.write("\n")

    # print('done!')

    # for f in sys.argv[1:]:
    for f in os.listdir(filepath):
        # print(sys.stderr, f)
        f = os.path.join(filepath, f)
        root = etree.parse(f).getroot()

        for contrib in root.iterfind(".//membercontribution"):
            # remove tables
            tables = contrib.findall(".//table")
            map(contrib.remove, tables)

            # find parent, which should always be a <p>
            p = contrib.getparent()
            assert p.tag == "p"
            # find adjacent <member>
            member = p.find("member")
            if member is not None:
                text = etree.tostring(member, encoding="unicode", method="text") + \
                    etree.tostring(contrib, encoding="unicode", method="text")
                constituency = member.find("memberconstituency")
                if constituency is not None:
                    constituency = constituency.text.strip("()")
                else:
                    constituency = ""
                member = member.text
            else:
                text = etree.tostring(contrib, encoding="unicode", method="text")
                member = ""
                constituency = ""

            # for speech acts wrapped in <debate><section> tags,
            # find parent to <p>, then find child <title>
            section = contrib.getparent().getparent()
            section2 = contrib.getparent().getparent().getparent()
            if section.tag == "section" and section2.tag != "section":
                # find child <title> tag
                title = section.find("title")
                if title is not None:
                    bill = etree.tostring(title, encoding="unicode", method="text")
                else:
                    bill = ""
                # find great great grandparent to <p>, which should be <houselords> or <housecommons>
                day = contrib.getparent().getparent().getparent().getparent()
                if day.tag == "houselords" or day.tag == "housecommons":
                    # find child <date> tag
                    date = day.find("date")
                    if date is not None:
                        date = date.attrib.get("format", "")
                    else:
                        date = ""
                else:
                    date = ""

            elif section.tag == "section" and section2.tag == "section":
                title = section.find("title")
                title2 = section2.find("title")
                if title is not None:  
                    bill = etree.tostring(title, encoding="unicode", method="text")
                elif title2 is not None:
                    bill = etree.tostring(title, encoding="unicode", method="text")
                else:
                    bill = ""
                day = contrib.getparent().getparent().getparent().getparent().getparent()
                if day.tag == "houselords" or day.tag == "housecommons":
                    date = day.find("date")
                    if date is not None:
                        date = date.attrib.get("format", "")
                    else:
                        date = ""
                else:
                    date = ""
                
            # for speech acts not wrapped in <debate><section> tags, get date
            else:
                house = contrib.getparent().getparent()
                bill = "" # bill title should always be "" because procedural language
                if house.tag == "houselords" or house.tag == "housecommons":
                    # find child <date> tag
                    date = house.find("date")
                    if date is not None:
                        date = date.attrib.get("format", "")
                    else:
                        date = ""
                else:
                    date = ""


            sys.stdout.write("\t".join(map(clean, (
                p.attrib.get("id", ""),
                date,
                bill,
                member,
                constituency,
                text))))

            sys.stdout.write("\n")

