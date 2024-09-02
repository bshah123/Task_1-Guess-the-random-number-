#include <iostream>
#include <cstdlib> // For rand(), srand()
#include <ctime>   // For time()

using namespace std;

int main() {
    // Seed the random number generator
    srand(static_cast<unsigned>(time(0)));
    
    // Generate a random number between 1 and 100
    int number_to_guess = rand() % 1000 + 1;
    
    int user_guess = 0;
    
    cout << "Welcome to the Guess the Number game!" << endl;
    cout << "I'm thinking of a number between 1 and 1000." << endl;
    
    while (user_guess != number_to_guess) {
        cout << "Enter your guess: ";
        cin >> user_guess;
        
        if (user_guess < number_to_guess ) {
            if(user_guess-number_to_guess>-50){
                cout<<"Somewhat low ! try higher"<<endl;
            }
            else cout << "Too low! Try again." << endl;
        } else if (user_guess > number_to_guess) {
            if(user_guess-number_to_guess<50){
                cout<<"Somewhat high ! try lower"<<endl;
            }
            else cout << "Too high! Try again." << endl;
        } else {
            cout << "Congratulations! You've guessed the number!" << endl;
        }
    }
    
    return 0;
}