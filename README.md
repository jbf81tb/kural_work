# kural_work

Machine learning work in PyTorch for the lab of Dr. Comert Kural (https://www.asc.ohio-state.edu/kural.1/).

Work in progress. Some tasks show performance, but none scaled into production yet.

* Movie RNN
    * Simplifying images
        * Sand-shaking algorithm
            * Perhaps implementing a signal-boosting algorithm (more below) would be preferrable
        * Cell outliner (__NN__)
            * Made use of K-means here
        * Crop cell to fixed-size image
        * Apply autoencoder (__NN__) to reduce dimentionality
    * Continuous-time RNN
        * Currently not enough training data taken into account.
        * Model not stable.
            * Predictions diverge or converge too quickly to get useful information.
    * Predict just one frame (__NN__)
        * Currently not enough training data taken into account.
        * Model is able to over-train and achieve desireable results, but unseen data produces unacceptable results.
* Signal-boosting with U-nets
    * Very easy to boost gross structure signal and flatten-out noise
    * Some fine detail is lost, but I believe building a better classifier (__NN__) and using perceptual loss can help this.