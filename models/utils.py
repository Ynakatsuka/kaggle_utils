def get_valid_and_test_predictions(cv, train, test, get_model_func, setting_name):
    n_splits = cv.n_splits

    val_scores = [0] * n_splits
    pred_valid = np.zeros((len(train), 1))
    pred_test = np.zeros((len(test), 1))

    X_test = get_X(test)

    for i, (fit_idx, val_idx) in enumerate(cv.split(train)):
        print('Training Fold {}...'.format(i+1))
        weight_path = './../data/' + setting_name + '_cv_' + str(i+1) + '.hdf5'

        X_fit = get_X(train.iloc[fit_idx])
        y_fit = get_y(train.iloc[fit_idx])
        X_val = get_X(train.iloc[val_idx])
        y_val = get_y(train.iloc[val_idx])

        model = get_model_func(numerical_features, categorical_features, train.columns, maxvalue_dict, maxlen, num_words, embedding_dims, embedding_matrix, seed)
        model.compile(loss='mse',
              optimizer=get_optimizer(),
              metrics=[rmse])
        callbacks = get_callbacks(weight_path)

        model.fit(X_fit, y_fit,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_val, y_val),
              callbacks=callbacks,
              verbose=0)
        model.load_weights(weight_path)

        pred_val = model.predict(X_val)
        pred_valid[val_idx] = pred_val
        val_scores[i] = np.sqrt(mean_squared_error(y_val, pred_val))
        pred_test += model.predict(X_test)/n_splits

        print('Fold {} RMSE: {:.5f}'.format(i+1, val_scores[i]))

        del X_val, X_fit, model; gc.collect(); K.clear_session()

    val_mean = np.mean(val_scores)
    val_std = np.std(val_scores)

    print('Local RMSE: {:.5f} (Â±{:.5f})'.format(val_mean, val_std))

    return pred_valid, pred_test, val_mean, val_std
