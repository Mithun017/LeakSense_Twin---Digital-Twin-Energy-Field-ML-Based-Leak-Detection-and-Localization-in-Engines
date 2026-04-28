import sys

with open('leak_sense_twin/main_leak_detection_system.py', 'r') as f:
    content = f.read()

old_evaluation_step = """    # Step 8: Evaluate on Test Set
    logger.info("Evaluating on test set...")
    # Convert to tensors for evaluation
    X_test_tensor = torch.FloatTensor(X_test)
    y_leak_test_tensor = torch.FloatTensor(y_leak_test)
    y_zone_test_tensor = torch.LongTensor(y_zone_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    leak_net = LeakSenseNet(input_dim=input_dim).to(device)
    loc_net = LeakLocalizationNet(input_dim=input_dim, n_classes=6).to(device)

    # Load best models (we would have saved them during training)
    # For demo, we'll just use the last epoch models

    leak_net.eval()
    loc_net.eval()

    with torch.no_grad():
        leak_outputs = leak_net(X_test_tensor.to(device))
        zone_outputs = loc_net(X_test_tensor.to(device))

        leak_preds = (leak_outputs > 0.5).float().cpu().numpy()
        _, zone_preds = torch.max(zone_outputs.data, 1)
        zone_preds = zone_preds.cpu().numpy()

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

        leak_accuracy = accuracy_score(y_leak_test, leak_preds)
        leak_precision = precision_score(y_leak_test, leak_preds, zero_division=0)
        leak_recall = recall_score(y_leak_test, leak_preds, zero_division=0)
        leak_f1 = f1_score(y_leak_test, leak_preds, zero_division=0)

        zone_accuracy = accuracy_score(y_zone_test, zone_preds)
        zone_precision = precision_score(y_zone_test, zone_preds, average='weighted', zero_division=0)
        zone_recall = recall_score(y_zone_test, zone_preds, average='weighted', zero_division=0)
        zone_f1 = f1_score(y_zone_test, zone_preds, average='weighted', zero_division=0)

    logger.info("=== TEST SET RESULTS ===")
    logger.info(f"Leak Detection:")
    logger.info(f"  Accuracy: {leak_accuracy:.4f}")
    logger.info(f"  Precision: {leak_precision:.4f}")
    logger.info(f"  Recall: {leak_recall:.4f}")
    logger.info(f"  F1-Score: {leak_f1:.4f}")
    logger.info(f"Zone Localization:")
    logger.info(f"  Accuracy: {zone_accuracy:.4f}")
    logger.info(f"  Precision: {zone_precision:.4f}")
    logger.info(f"  Recall: {zone_recall:.4f}")
    logger.info(f"  F1-Score: {zone_f1:.4f}")"""

new_evaluation_step = """    # Step 8: Evaluate on Test Set
    logger.info("Evaluating on test set...")
    # Use the trained networks from the ensemble
    leak_net = ensemble.leak_net
    loc_net = ensemble.loc_net

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    leak_net.to(device)
    loc_net.to(device)

    leak_net.eval()
    loc_net.eval()

    # Convert to tensors for evaluation
    X_test_tensor = torch.FloatTensor(X_test)
    y_leak_test_tensor = torch.FloatTensor(y_leak_test)
    y_zone_test_tensor = torch.LongTensor(y_zone_test)

    with torch.no_grad():
        leak_outputs = leak_net(X_test_tensor.to(device))
        zone_outputs = loc_net(X_test_tensor.to(device))

        # Debug: print statistics of model outputs
        logger.info(f"Leak output stats - min: {leak_outputs.min():.4f}, max: {leak_outputs.max():.4f}, mean: {leak_outputs.mean():.4f}")
        logger.info(f"Leak output percentiles - 10th: {torch.quantile(leak_outputs, 0.1):.4f}, 50th: {torch.quantile(leak_outputs, 0.5):.4f}, 90th: {torch.quantile(leak_outputs, 0.9):.4f}")

        leak_preds = (leak_outputs > 0.5).float().cpu().numpy()
        _, zone_preds = torch.max(zone_outputs.data, 1)
        zone_preds = zone_preds.cpu().numpy()

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

        leak_accuracy = accuracy_score(y_leak_test, leak_preds)
        leak_precision = precision_score(y_leak_test, leak_preds, zero_division=0)
        leak_recall = recall_score(y_leak_test, leak_preds, zero_division=0)
        leak_f1 = f1_score(y_leak_test, leak_preds, zero_division=0)

        zone_accuracy = accuracy_score(y_zone_test, zone_preds)
        zone_precision = precision_score(y_zone_test, zone_preds, average='weighted', zero_division=0)
        zone_recall = recall_score(y_zone_test, zone_preds, average='weighted', zero_division=0)
        zone_f1 = f1_score(y_zone_test, zone_preds, average='weighted', zero_division=0)

    logger.info("=== TEST SET RESULTS ===")
    logger.info(f"Leak Detection:")
    logger.info(f"  Accuracy: {leak_accuracy:.4f}")
    logger.info(f"  Precision: {leak_precision:.4f}")
    logger.info(f"  Recall: {leak_recall:.4f}")
    logger.info(f"  F1-Score: {leak_f1:.4f}")
    logger.info(f"Zone Localization:")
    logger.info(f"  Accuracy: {zone_accuracy:.4f}")
    logger.info(f"  Precision: {zone_precision:.4f}")
    logger.info(f"  Recall: {zone_recall:.4f}")
    logger.info(f"  F1-Score: {zone_f1:.4f}")"""

if old_evaluation_step in content:
    content = content.replace(old_evaluation_step, new_evaluation_step)
    with open('leak_sense_twin/main_leak_detection_system.py', 'w') as f:
        f.write(content)
    print("Replacement successful.")
else:
    print("Old evaluation step not found. Please check the file.")
    # Let's try to find a similar string by looking for the start and end markers.
    # We'll do a more flexible replacement by splitting the content.
    import re
    # Pattern to match the evaluation step from the comment to the line before the saving models comment.
    pattern = r'(\s*# Step 8: Evaluate on Test Set\s*logger\.info\("Evaluating on test set\.\.\.".*?logger\.info\(f"Zone Localization:\s*\))'
    # This is complex, so we'll just print the first 2000 characters of the file to see what we have.
    print("First 2000 characters of the file:")
    print(content[:2000])