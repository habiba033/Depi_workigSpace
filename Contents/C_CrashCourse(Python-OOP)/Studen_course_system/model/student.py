'''
doc

'''
class Student:
    _id_counter = 1
    def __init__(self,name):
        self.name = name
        self.student_id = Student._id_counter
        Student._id_counter+=1
        self.grades = {}
        self.enrolled_courses = []
        
    def __str__(self):
        return f"student ID : {self.student_id} , Name : {self.name} , grades : {self.grades} , enrolled in :{self.enrolled_courses}"
    def __repr__(self):
        return f"student ID : {self.student_id} , Name : {self.name} , grades : {self.grades} , enrolled in :{self.enrolled_courses}"
    
    def add_grade(self,course_id,grade):
        self.grades[course_id] = grade
    
    def enrolled_in_course(self,course):
        self.enrolled_courses.append(course)
