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



