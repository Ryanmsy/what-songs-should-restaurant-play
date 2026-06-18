how Linear Algebra ties into recommendion system. 

I did not learn about linear algebra and hence this is my time using gilbert strang course.

I think the 4 subspace is important to mention here. 

Null Space of RM, Null Space of RN Row Space, Column Space


Column Space and its reatalbe to data science 

It likes within R M(rows) because within one column, it will only go trhough m rows hence it contains exactly m components 

What does it tell us: The columns tell us the features, if we have 10 columns like Binary Tasty (1,0), Loud (1,0), Happy Hour (1,0)
Then the Columns tell us there are 10 features. Doing RREF on matrix / Dataset A will tell us which columns are pivot (independent and necessary for analysis)

For example, we have 10 columns 1 to 10, if after RREF, we get Columns 2 (loudness),5 (happyhour),6 (outdoor seating) with numbers, we know those columns are pivot and basis of the column space ( idenpendent), the rest are reduendant. Instead of doing analysis for all, we know only to use those 3 columns.  

It tells us how our obersvation (rows) move compared to other features 




Row Space

It exist within R N because if we look at one row, it can only have the the count of how many columns there are 





Null Space
Ax = 0 

A  = original dataset 

x = weights applied to A 

what weights applied to A will make it zero? 

It is important because if there is a vector tht exist within the null space of A, that is telling us there is column that is redundant and needs to be remov



--- add about left null space 

-- add about rank 

-- add about dot roduct 
A dot product is a single value that tells us how much two vectors align or point in the same direction. 


|A| = magnitude (length) of the vector

Cosine tell us horiozontla movement - how far left or right did we move 

Sine tell us vertical movement - how far forward or backward did we move 


Cosine Similiarity
we use consine instead of sine because when it is -180, cosine will show it is negative correalated while sine will show it is 0 so no differene 


Fun Fact: 
it is using in many video games like determining if someone is front or back, where the shadow should be (rendereing light equation)



-- matrixp roduct 



-- PCA 
