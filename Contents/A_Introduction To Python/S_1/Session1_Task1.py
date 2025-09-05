'''
Task 1: YOU HAVE THIS EMAIL ADDRESS :"Amit_ml@gmail.com"
        1. input validation : check if this email address contains one "@" and at least one "." after "@" 
        if not print "Invalid email address"
        2. extract username and domain name from this email address and print them
        3. check for domain ending : if domain name ends with ".com" print "Commercial domain"
            if domain name ends with ".edu" print "educational domain"
            if domain name ends with ".org" print "Organizational domain"
            else print "Other domain" 
            '''
email = "Amit_ml@gmail.com"
# 1- input validation
if "@" in email and '.' in email:
    print('Vaild')
else:
    print("Invalid email address")

# 2- extract username and domain name
at_idx = email.index('@')
username = email[:at_idx]
print(username)

domain = email[at_idx+1:]
print(domain)

#3- cheching the domain
if ".com" in domain :
    print("commercial domain")
elif ".edu" in domain:
    print("educational domain")
elif ".org" in domain:
    print("organizational domain")
else:
    print("other domain")
    
# 4- cheching the user name
