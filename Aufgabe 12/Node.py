
import sys

def get_label(string, start):
   ''' scans the next category symbol or word '''
   for end in range(start, len(string)):
      if string[end] in ' ()[]':
         return string[start:end], end
   sys.exit("Error: bad input format at: "+string)

   
class Node(object):
   ''' class for parse tree nodes '''
   
   def __init__(self, string, spos=0, wpos=0):
      self.start = self.end = wpos
      if string[spos] == '(':
         self.label, spos = get_label(string, spos+1)
         self.children = []
         while string[spos] != ')':
            child = Node(string, spos, self.end)
            self.children.append(child)
            spos = child.endpos
            self.end = child.end
         spos += 1
      elif string[spos] == ' ':
         self.label, spos = get_label(string, spos+1)
         self.end = self.start+1
      else:
         sys.exit("Input error at: "+string[spos:])
      self.endpos = spos

   def is_final(self):
      return not hasattr(self, "children")

   def __str__(self):
      if self.is_final():
         return " "+self.label
      child_strings = (str(child) for child in self.children)
      return "(" + self.label + ''.join(child_strings) +")"

   def words_and_constituents(self):
      ''' return the list of words and the list of constituents for this parse '''
      if self.is_final():
         return [self.label], []
      
      if len(self.children) == 1 and not self.children[0].is_final():
         w, c = self.children[0].words_and_constituents()
         label, start, end = c[0]
         # replace chain rule with a complex label
         c[0] = (self.label+' '+label, start, end)
         return w,c
      
      words = []
      constituents = [(self.label, self.start, self.end)]
      for child in self.children:
         w, c = child.words_and_constituents()
         words += w
         constituents += c
         
      return words, constituents

