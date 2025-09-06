class Student:
    #attributes
    name= "Habiba"
    age=22
    gender ='female'
    gpa = 3.3
    #methods
    def info(self):
        '''
        print the student information
        '''
        print("hello")  

# #create object - instant - copy from the class
# s1 = Student()
# s1.name='hagar'
# print(s1.name)
# print(s1.info())


class Test:
    def info(self):
        print(self)
        
x=Test()
print(x)
x.info()