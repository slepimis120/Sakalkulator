from audio_calculator import test_audio
from helper.audio_capturer import record_voice, save_recording
from helper.video_capturer import record_video
from CNN import get_model
from SVM import get_audio_model
from video_calculator import test_video


def meni():
    print("-"*20 + "Meni" + 20*"-")
    print("1. Istreniraj model za klasifikaciju slika (CNN)")
    print("2. Istreniraj model za klasifikaciju zvuka (SVM)")
    print("3. Testiraj rad kalkulatora nad video skupom")
    print("4. Testiraj rad kalkulatora nad audio skupom")
    print("5. Snimi video i saznaj odgovor Sakalkulatora!")
    print("6. Snimi zvuk i saznaj odgovor Sakalkulatora!")
    print("7. Bye Sakalkulatoru :( ")

    while True:
        user_input = input()
        if user_input == '1':
            meni_1()
        elif user_input == '2':
            meni_2()
        elif user_input == '3':
            meni_3()
        elif user_input == '4':
            meni_4()
        elif user_input == '5':
            meni_5()
        elif user_input == '6':
            meni_6()
        elif user_input == '7':
            print("Do idućeg druženja !")
            break


def meni_1():
    print("Treniranje slika počelo...")

    # TODO: path preuzeti kao parametar
    path = './data/video/training_data'
    model = get_model(path)

    print("Završeno treniranje!")
    print("Vraćeni ste na glavni meni")


def meni_2():
    print("Treniranje zvuka počelo...")

    # todo zovni funkciju za treniranje
    path = './data/audio/training_data'
    get_audio_model(path)

    print("Završeno treniranje!")
    print("Vraćeni ste na glavni meni")


def meni_3():
    print("Testiranje videa počelo...")

    test_video("data/video/testing_data/")

    print("Završeno testiranje!")
    print("Vraćeni ste na glavni meni")


def meni_4():
    print("Testiranje zvuka počelo...")

    test_audio("data/audio/testing_data/")

    print("Završeno testiranje!")
    print("Vraćeni ste na glavni meni")


def meni_5():
    record_video()
    # todo obradi ovo sto se snimilo i prikazi rezultat
    print("Vraćeni ste na glavni meni")


def meni_6():
    captured_audio = record_voice()
    save_recording(captured_audio)
    # todo obradi ovo sto se snimilo i prikazi rezultat
    print("Vraćeni ste na glavni meni")


if __name__ == '__main__':
    meni()
