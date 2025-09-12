from ..model.course import Course

from ..model.student import Student




class SystemManager:
    def __init__(self):
        self.students = {}
        self.courses = {}
        
    # student as instane but in shape of function
    def add_student(self,name):
        student = Student(name)
        self.students[student.student_id] = student
        print("student added successfully !")
        return student.student_id
    
    def remove_student(self,student_id):
        if student_id in self.students:
            student = self.students[student_id]
            if not student.enrolled_courses:
                del self.students[student_id]
                print("student removed successfully !")
            else:
                print("student is enrolled in courses, cannot remove!")
        else:
            print("student not found !")
            
    def enroll_course(self,student_id,course_id ):
        if student_id in self.students and course_id in self.courses:
            student = self.students[student_id]
            course = self.courses[course_id]
            if course.name not in student.enrolled_course:
                course.enroll_student(student)
                student.enrolled_in_course(course)
                print("student enrolled in course successfully !")
        else:
            print("student or course not found !")
            
    def record_grade(self,student_id,course_id,grade):
        if student_id in self.students and course_id in self.courses:
            student = self.students[student_id]
            course = self.courses[course_id]
            student.add_grade(course_id,grade)
            print("grade recorded successfully !")
        else:
            print("Invalid Student ID or Course ID")
            
    def get_all_students(self):
        return list(self.students.values())
    
    def get_all_courses(self):
        return list(self.courses.values())