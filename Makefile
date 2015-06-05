cond-phrase: cond-phrase.cc
	g++ -std=c++11 -Wall    -DDEBUG -D_DEBUG -g -rdynamic -I/home/austinma/git/cpyp/ cond-phrase.cc -o cond-phrase -lboost_program_options-mt -lboost_serialization
#	g++ -std=c++11 -Wall -O                               -I/home/austinma/git/cpyp/ cond-phrase.cc -o cond-phrase -lboost_program_options-mt -lboost_serialization
