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

'''
doc ...

'''
class Course:
    _id_counter = 1
    
    def __init__(self,name):
        self.name = name
        self.course_id = Course._id_counter
        Course._id_counter += 1
        self.enrolled_students = []
        
    def __str__(self):
        return f"course ID : {self.course_id} , Name : {self.name} , Enrolled : {len(self.enrolled_students)}"
    
    def enroll_student(self,student):
        if student not in self.enrolled_students:
            self.enrolled_students.append(student)
            print("student enrolled successfully !")
        else :
            print("student already enrolled")
            
    def remove_student(self, student):
        if student in self.enrolled_students:
            self.enrolled_students.remove(student)
            print("student removed successfully!")
        else:
            print("student not found in this course")
            
            