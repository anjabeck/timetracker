# Time tracker
This repository allows to visualize tracked times. The activities are read from a google sheets document with the columns
* Activity (short description, e.g. oil painting dog). This entry is not yet used for the most part.
* Date (the date of the activity)
* Start and End (the start and end time of the activity)
* Duration (this is used as a cross-check for start and end)
* Category (The category of the activity. This could be anything. Currently, the only hardcoded activity is "Reading" as there is a specific extra figure for it.)
* Category 2 (A secondary category, e.g. singing in a choir might be Socializing and Music. This is not yet used but might be added in the future.)

The `sheets_id.txt` file needs to be filled with the google sheets ID. The sheet must be made readable for people with the shared link.

The only constraint right now is the words-per-minute figure. It is produced using the category "Reading". The Activity entries for this category need to be the book titles. For each book, there needs to be a corresponding entry in `books.csv` with an estimated wordcount. The ones in the example are from [howlongtoread.com/books]. Moreover, a `jpg`file with each book's cover should be deposited in `book-covers`.

## Colours
The beautiful canyon colours are taken from [kevinsblake/NatParksPalettes](https://github.com/kevinsblake/NatParksPalettes/blob/main/R/NatParksPalettes.R).

## Potential future improvements:
* Add some basic NLP to deal with potential typos in the activity and category.
* Add Category 2 information.

## Example figures
Activity types removed for privacy.

![Activites per Week](plots/activity_times_week.pdf)
![Words per Minute](plots/reading_wpm.pdf)