function ax = subplotsquare(m, i)
%subplotsquare create and return subplot axes index i out of a total of m
%subplots such that they are all arranged approximately squarely.
columns = ceil(sqrt(m));
rows = ceil(m / columns);
ax = subplot(rows, columns, i);
end