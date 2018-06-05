import csv
temp = []
protocol_type_dict={'tcp':1, 'udp':2, 'icmp':3}


with open('KDDTest+.csv', newline='') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		rows = {'duration':row[0], 'protocol_type':row[1], 'service':row[2],
				'flag':row[3], 'src_bytes':row[4], 'dst_bytes':row[5],
				'land':row[6], 'wrong_fragment':row[7], 'urgent':row[8],
				'hot':row[9], 'num_failed_logins':row[10], 'logged_in':row[11],
				'num_compromised':row[12], 'root_shell':row[13], 'su_attempted':row[14],
				'num_root':row[15], 'num_file_creations':row[16], 'num_shells':row[17],
				'num_access_files':row[18], 'num_outbound_cms':row[19], 'is_host_login':row[20],
				'is_guest_login':row[21], 'count':row[22], 'srv_count':row[23], 'serror_rate':row[24],
				'srv_serror_rate':row[25],'rerror_rate':row[26], 'srv_rerror_rate':row[27],
				'same_srv_rate':row[28], 'diff_srv_rate':row[29], 'srv_diff_host_rate':row[30],
				'dst_host_count':row[31],'dst_host_srv_count':row[32], 'dst_host_same_srv_rate':row[33],
				'dst_host_diff_srv_rate':row[34], 'dst_host_same_src_port_rate':row[35],
				'dst_host_srv_diff_host_rate':row[36], 'dst_host_serror_rate':row[37],
				'dst_host_srv_serror_rate':row[38], 'dst_host_rerror_rate':row[39],'dst_host_srv_rerror_rate':row[40], 'target':row[41]}
		temp.append(rows)
		print(row[1])
with open('test.csv', 'w', newline='') as csvfile:
    fieldnames = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    				'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    				'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    				'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cms',
    				'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 
    				'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
    				'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    				'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    				'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'target']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()


    for each in temp:
    		#writer.writerow(each)
    		if each['protocol_type'] == 'tcp':
    			each['protocol_type'] = protocol_type_dict.get('tcp')
    		elif each['protocol_type'] == 'udp':
    			each['protocol_type'] = protocol_type_dict.get('udp')
    		elif each['protocol_type'] == 'icmp':
    			each['protocol_type'] = protocol_type_dict.get('icmp')
		
    		if each['target'] == 'normal':
    			each['target'] = 0
    		else:
    			each['target'] = 1

    		writer.writerow(each)