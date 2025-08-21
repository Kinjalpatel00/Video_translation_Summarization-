
# import mysql.connector
 
# mydb = mysql.connector.connect(
#     host = "localhost",
#     user = "root",
#     password = ""
# )
 
# # Creating an instance of 'cursor' class 
# # which is used to execute the 'SQL' 
# # statements in 'Python'
# cursor = mydb.cursor()
 
# # Creating a database with a name
# # 'geeksforgeeks' execute() method 
# # is used to compile a SQL statement
# # below statement is used to create 
# cursor.execute("CREATE DATABASE python_database")


# import mysql.connector

# mydb = mysql.connector.connect(
# 	host = "localhost",
# 	user = "root",
# 	password = "",
# 	database = "python_database"
# )

# cursor = mydb.cursor()

# cursor.execute("CREATE TABLE testing (ID INT PRIMARY KEY, name VARCHAR(255), user_name VARCHAR(255))")


import mysql.connector

# Connect to the MySQL server
db = mysql.connector.connect(
host="localhost",
user="root",
password="",
database="python_database"
)

# Create a cursor object
cursor = db.cursor()

insert_query = """
        INSERT INTO testing (ID, name, user_name) VALUES (%s, %s, %s)
        """
        
# Data to be inserted
data = [
        (2, 'John Doe', 'johndoe'),
        (3, 'Jane Smith', 'janesmith'),
        (4, 'Alice Johnson', 'alicejohnson')
    ]
        
# Execute insert query
cursor.executemany(insert_query, data)
        

# Commit the changes
db.commit()

# Print the number of rows affected
print(cursor.rowcount, "record inserted.")
