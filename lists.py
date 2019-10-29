# Import the `collections` library
import collections
# Import `choice` from the `random` library
from random import choice
from random import randrange



#lists - tuples - dictionaries - sets
# These list elements are all of the same type
zoo = ['bear', 'lion', 'panda', 'zebra']
print(zoo)

# But these list elements are not
biggerZoo = ['bear', 'lion', 'panda', 'zebra', ['chimpanzees', 'gorillas', 'orangutans', 'gibbons']]
print(biggerZoo)

oneZooAnimal = biggerZoo[0]

# Print `oneZooAnimal`
print(oneZooAnimal)

monkeys = biggerZoo[-1]
print(monkeys)

# Pass -2 to the index operator on biggerZoo
zebra = biggerZoo[-2]
print(zebra)

#tuples:
#tuples are write protected and faster than lists to iterate

#Dictionaries:
#Dictionaries are known to associate each key with a value, while lists just contain values.
#Use a dictionary when you have an unordered set of unique keys that map to values.
#Note that, because you have keys and values that link to each other, the performance will be better than lists in cases where you’re checking membership of an element.

#sets: You should make use of sets when you have an unordered set of unique, immutable values that are hashable.

# Check if a dictionary is hashable
print(isinstance({}, collections.Hashable))

# Check if a float is hashable
print(isinstance(0.125, collections.Hashable))

print(biggerZoo)
# Use the slice notation like this
someZooAnimals = biggerZoo[2: ]

# Print to see what you exactly select from `biggerZoo`
print(someZooAnimals)

# Try putting 2 on the other side of the colon
otherZooAnimals = biggerZoo[:2]

# Print to see what you're getting back
print(otherZooAnimals)

print(biggerZoo[2::2])
print(biggerZoo[1::3])

# Construct your `list` variable with a list of the first 4 letters of the alphabet
list = ['a', 'b', 'c', 'd']

# Print your random 'list' element
print(choice(list))

# Construct your `randomLetters` variable with a list of the first 4 letters of the alphabet
randomLetters = ['a', 'b', 'c', 'd']

# Select a random index from 'randomLetters`
randomIndex = randrange(0 , len(randomLetters))
#randomIndex = randrange(0 , 4)

# Print your random element from `random`
print(randomLetters[randomIndex])




shortList = ['a', 'b', 'c', 'd']
longerList = ['e', 'f', 'g', 'h']
# Append [4,5] to `shortList`
shortList.append([4, 5])

# Use the `print()` method to show `shortList`
print(shortList)

# Extend `longerList` with [4,5]
longerList.extend([4, 5])

# Use the `print()` method to see `longerList`
print(longerList)

rooms = ['d', 'f', 'j', 'a']
orders = ['z', 'v', 'c', 'b']

# Use `sort()` on the `rooms` list
rooms.sort()

# Print out `rooms` to see the result
print(rooms)

# Now use the `sorted()` function on the `orders` list
sorted(orders)

# Print out orders
print(orders)














# This is your list
objectList = ['v','b',['ab','ba']]

# Copy the `objectList`
copiedList = objectList[:]

# Change the first list element of `copiedList`
copiedList[0] = 'c'

# Go to the third element (the nested list) and change the second element
copiedList[2][1] = 'd'

# Print out the original list to see what happened to it
print(objectList)



myList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
[(lambda x: x*x)(x) for x in myList]

print('myList', myList)



# Your initial list of lists
list = [[1,2],[3,4],[5,6]]
print('list ',list)
# Flatten out your original list of lists with `sum()`
sum(list, [])

print('list2 ',list)





# Your list with duplicate values
duplicates = [1, 2, 3, 1, 2, 5, 6, 7, 8]

# Print the unique `duplicates` list
print( list(set(duplicates)) )
