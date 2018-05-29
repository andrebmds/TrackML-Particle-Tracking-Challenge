import numpy as np 
import pandas as pd 
import json

dna_json = "{\"deviceSignature\":{\"collector\": \"iOS\",\"collectorVersion\": \"8.1.2.2\",\"browser\":{\"userAgent\": \"Mozilla 5.0 (iPhone; CPU iPhone OS 10_3_3 like Mac OS X) AppleWebKit 603.3.8 (KHTML, like Gecko) Mobile 14G60\",\"vendor\": \"Apple Inc.\",\"vendorSubId\": \"iPhone8,2\",\"buildId\": \"14G60\",\"cookieEnabled\": null},\"location\":{\"latitude\": \"-25.470688\",\"longitude\": \"-49.285069\"},\"telephony\":{\"carrierName\": \"VIVO\",\"carrierCountry\": \"BR\",\"carrierMobileCountryCode\": null,\"carrierIsoCountryCode\": null,\"carrierMobileNetworkCode\": null,\"carrierAllowsVoip\": true,\"externalIpAddress\": null,\"ipAddress\": null,\"macAddress\": \"00:00:00:00:00:00\",\"netmaskAddress\": null,\"broadcastAddress\": null,\"connected\": false},\"screen\":{\"height\": \"736 Pixels\",\"width\": \"414 Pixels\",\"orientation\": \"1\"},\"system\":{\"platform\": \"iPhone\",\"deviceName\": \"iPhone\",\"systemName\": \"iOS\",\"osVersion\": \"10.3.3\",\"systemDeviceTypeUnformatted\": \"iPhone8,2\",\"systemDeviceTypeFormatted\": \"iPhone 6s Plus\",\"multitaskingEnabled\": true,\"proximitySensor\": true,\"debuggerAttached\": false,\"jailBroken\": false,\"numberOfProcessors\": \"2\",\"numberOfActiveProcessors\": \"2\",\"processorBusSpeed\": \"0Mhz\",\"accessoriesAttached\": false,\"numberOfAttachedAccessories\": \"0\",\"nameOfAttachedAccessories\": null,\"headphoneAttached\": false,\"locale\": \"pt_BR\",\"language\": \"pt-BR\",\"timeZone\": \"America/Sao_Paulo\",\"currency\": \"R$\",\"measurementSystem\": \"Metric\",\"applicationVersion\": \"2\",\"vendorId\": \"CB89C590-2E9D-413C-8483-23FC11A3B7E4\",\"totalDiskSpace\": \"59.59 GB\",\"totalMemory\": \"2048.000000\",\"processName\": \"Next-Bradesco_Si\"},\"wifi\":{\"ipAddress\": null,\"macAddress\": \"02:00:00:00:00:00\",\"netmaskAddress\": null,\"broadcastAddress\": null,\"routerAddress\": \"127.0.0.1\",\"connected\": false}},\"ipAddress\": null}"

dna = json.loads(dna_json)

# Transform dna to DataFrame
dna_dict = {}
def readDNA(file_read, key=''):
	if isinstance(file_read,dict):
		[readDNA(file_read[key], key) for key in file_read.keys()]	
	else:
		print('{} : {} => {}'.format(key,file_read, type(file_read)))
		dna_dict[key] = file_read

readDNA(dna)
print(dna_dict)
df = pd.DataFrame([dna_dict])

print(df.info())

print('---------')

row_dna = {}
def populate_df_dna():
	key = 'file'
	if isinstance(st, object):
		pd.util.testing.rands(10)		

print(pd.util.testing.rands(10))