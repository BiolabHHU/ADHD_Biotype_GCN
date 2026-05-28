
import tensorflow as tf
from model_GCN import VAE_ADHD_subtype,load_ADHD_data,loss_function_2,loss_function_diff,loss_function_diff2


if __name__ == '__main__':
    K_list = []
    NUM = 50
    num_K = 3 # biotypes + HC
    num_epochs = 500

    for K in range(NUM):

        FC_final, ALFF_final, label_final = load_ADHD_data()
        source_label = label_final

        vae_ADHD = VAE_ADHD_subtype()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


        for epoch in range(num_epochs):
            Flag = 0
            if epoch == 0:
                Flag = 1
                mu_ave = tf.zeros([num_K, 20], dtype=tf.float32)#初始

            with tf.GradientTape() as tape:

                reconstructed_x, reconstructed_fc,mu, logvar, class_output, closest_idx_all, mu_ave,z \
                    = vae_ADHD.forward_1(alff=ALFF_final,fc=FC_final, flag=Flag, mu_ave=mu_ave, label=source_label)

                MSE, KLD, CE =loss_function_2(reconstructed_x=reconstructed_x,
                                    reconstructed_fc=reconstructed_fc,
                                    alff=ALFF_final,
                                    fc = FC_final,
                                    label=closest_idx_all,
                                    class_output=class_output,
                                    mu=mu,
                                    logvar=logvar)
                MU, _ = loss_function_diff(mu_ave)
                reconstructed_mu_ave = vae_ADHD.decoder.call(mu_ave,tf.math.unsorted_segment_mean(FC_final, closest_idx_all, num_segments=3))
                reconstructed_mu_ave = tf.reshape(reconstructed_mu_ave, (reconstructed_mu_ave.shape[0], -1))
                Recon_MU,Recon_MU_mse = loss_function_diff(reconstructed_mu_ave)

                loss_total = 100 * MSE + 1 * KLD + 10000 * CE + 10000 * MU + 1 * Recon_MU - 1 * Recon_MU_mse

            grads = tape.gradient(loss_total, vae_ADHD.trainable_variables)
            optimizer.apply_gradients(zip(grads, vae_ADHD.trainable_variables))

            print(f'K:{K + 1}/{1},epoch:{epoch+1}/{num_epochs},MSE:{MSE:.5f},CE:{CE:.5f},MU:{MU:.5f}')






