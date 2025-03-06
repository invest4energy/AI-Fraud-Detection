# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy:.4f}')

# Print classification report
print(classification_report(y_test, y_pred))

# Document the performance
with open('model_evaluation_report.txt', 'w') as file:
    report = classification_report(y_test, y_pred)
    file.write(f'Test Accuracy: {accuracy:.4f}\n\n')
    file.write(report)
    
print("Model evaluation report saved as 'model_evaluation_report.txt'.")
