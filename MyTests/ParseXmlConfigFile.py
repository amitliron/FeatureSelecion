import xml.etree.ElementTree as ET


def main():
    xml_string = "C:\ELK_TUT\PythonWS\FeatureSelecion\MyTests\settings.xml"
    tree = ET.parse(xml_string)
    root = tree.getroot()

    # GENERAL = 0
    # LowVariance = 1
    # FilterMethods = 2
    # WrapperMethods = 3
    #
    # print(root[GENERAL].attrib.get('wrapperMethod'))
    # print(root[FilterMethods][0].attrib.get('Method'))
    # print(root[FilterMethods][0].attrib.get('Method'))

    # for neighbor in root.iter('neighbor'):
    #     print(neighbor.attrib)
    #
    for child in root:
        if child.tag=="General":

            print("EnableFeatureSelection = ", child.attrib.get('FeatureSelection'))
            print("EnableLowVarianceMethod =", child.attrib.get('LowVarianceMethod'))
            print("EnableFilterMethod = ", child.attrib.get('FilterMethod'))
            print("EnableWrapperMethod = ", child.attrib.get('WrapperMethod'))
            print("DebugTestAllOptions = ", child.attrib.get('DebugTestAllOptions'))

        elif child.tag=="LowVariance":
            print("LowVariance = ", child[0].attrib.get('value'))
        elif child.tag == "FilterMethods":
            print("FilterMethods = ", child[0].attrib.get('Method'))
        elif child.tag == "WrapperMethods":
            print("WrapperMethods = ", child[0].attrib.get('Method'))
            print("WrapperMethods = ", child[0].attrib.get('RunInBackground'))
        elif child.tag == "HybridMethod":
            print("HybridMethod = ", child[0].attrib.get('HybridMethod'))

    # print(root[0].attrib["Enable"])
    # print(root[0].attrib["LowVarianceMethod"])
    # print(root[0].attrib["FilterMethod"])
    # print(root[0].attrib["wrapperMethod"])

    # from xml.dom import minidom
    # doc = minidom.parse(xml_string)
    #
    # print(doc.childNodes[0].nodeValue)
    # print(doc.getElementsByTagName("General").childNodes[0])
    #
    # general = doc.getElementsByTagName("General")
    # print(general.length)




if __name__ == "__main__":
    main()