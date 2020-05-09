import xml.etree.ElementTree as ET

class FeatureSelectionConfigurations:

    def __init__(self,filePath):
        self._read_configuration(filePath)

    def _read_configuration(self, filePath):
        tree = ET.parse(filePath)
        root = tree.getroot()
        for child in root:
            if child.tag == "General":

                self.EnableFeatureSelection = child.attrib.get('EnableFeatureSelection').lower()
                self.DebugTestAllOptions = child.attrib.get('DebugTestAllOptions').lower()
                self.EnableLowVarianceMethod = child.attrib.get('LowVarianceMethod').lower()
                self.EnableFilterMethod = child.attrib.get('FilterMethod').lower()
                self.EnableWrapperMethod = child.attrib.get('WrapperMethod').lower()
                self.EmbeddedMethod = child.attrib.get('EmbeddedMethod').lower()

            elif child.tag == "LowVariance":
                self.LowVariance_value = child[0].attrib.get('value').lower()
            elif child.tag == "FilterMethods":
                self.FilterMethods_Method = child[0].attrib.get('Method').lower()
                filter_methods_types = ["pca"]
                if self.FilterMethods_Method not in filter_methods_types:
                    print("[ERROR] FilterMethods doesn't contain: ", self.FilterMethods_Method)
                    self.FilterMethods_Method = filter_methods_types[0]

            elif child.tag == "WrapperMethods":
                self.WrapperMethods_method = child[0].attrib.get('Method').lower()
                wrapper_methods_method_types = ["sfs", "sbs", "sffs", "sbfs", "rfecv"]
                if self.WrapperMethods_method not in wrapper_methods_method_types:
                    print("[ERROR] WrapperMethods doesn't contain: ", self.WrapperMethods_method)
                    self.WrapperMethods_method = wrapper_methods_method_types[0]
                self.WrapperMethods_RunInBackground = child[0].attrib.get('RunInBackground').lower()
                WrapperMethods_Scoring_types = ["accuracy", "roc_auc"]
                self.WrapperMethods_Scoring = child[0].attrib.get('Scoring').lower()
                if self.WrapperMethods_Scoring not in WrapperMethods_Scoring_types:
                    print("[ERROR] WrapperMethods_method doesn't contain: ", self.WrapperMethods_Scoring)
                    self.WrapperMethods_Scoring = WrapperMethods_Scoring_types[0]
            elif child.tag == "EmbeddedMethod":
                self.EmbeddedMethod_Method = child[0].attrib.get('Method').lower()
                embedded_method_types = ["rf", "lasso"]
                if self.EmbeddedMethod_Method not in embedded_method_types:
                    print("[ERROR] EmbeddedMethod doesn't contain: ", self.EmbeddedMethod_Method)
                    self.EmbeddedMethod_Method = embedded_method_types[0]


def main():
    xml_string = "C:\ELK_TUT\PythonWS\FeatureSelecion\MyTests\settings.xml"
    fss = FeatureSelectionConfigurations(xml_string)
    print(fss.EnableFeatureSelection)
    print(fss.FilterMethods_Method)
    print(fss.WrapperMethods_RunInBackground)
    print(fss.WrapperMethods_Scoring)
    print(fss.LowVariance_value)




if __name__ == "__main__":
    main()