""" 
    reshape tensor to have the tail dimensions only
    e.g. (1, 20, 30) -> (20, 30)
"""
tail_reshape(x) = reshape(x, Base.tail(size(x)))

# round to 2 decimal places for logging
round2(x)=round(x, digits = 2)
