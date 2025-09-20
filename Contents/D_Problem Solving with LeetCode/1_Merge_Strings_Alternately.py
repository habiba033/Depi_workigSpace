"""
You are given two strings word1 and word2. Merge the strings by adding letters in alternating order, starting with word1.
If a string is longer than the other, append the additional letters onto the end of the merged string.

Return the merged string."""

word1 = input("enter the first word :")
word2 = input("enter the second word :")
i=0
for i in range(len(word1)):
    print("!")