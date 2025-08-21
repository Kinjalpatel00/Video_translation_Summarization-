# Importing module 
import mysql.connector

# Creating connection object
mydb = mysql.connector.connect(
	host = "localhost",
	user = "root",
	password = ""
)

# Printing the connecti
# on object 
print(mydb)
