if [ $2 = "pipe.py" ];
then

	if [ $1 = "t" ];
	 then
		echo "Listing the trace of function calls"
		python3 -m trace --trace pipe.py

	elif [ $1 = "l" ];
        then
		echo "Listing only the functions "
		python3 -m trace --listfunc pipe.py
	else
		echo "Wrong choice"
	
	fi	
else
	echo "script missing"
fi
