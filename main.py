from helper.audio_capture import record_voice, save_recording
from helper.video_capture import record_video


def meni():
    print("-"*20 + "Meni" + 20*"-")
    print("1. Istreniraj model za klasifikaciju slika (CNN)")
    print("2. Istreniraj model za klasifikaciju zvuka (SVM)")
    print("3. Testiraj rad kalkulatora nad skupom videa")
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
            break


def meni_1():
    print("Treniranje slika počelo...")

    # todo zovni funkciju za treniranje

    print("Završeno treniranje!")


def meni_2():
    print("Treniranje zvuka počelo...")

    # todo zovni funkciju za treniranje

    print("Završeno treniranje!")


def meni_3():
    print("Testiranje videa počelo...")

    # todo zovni funkciju za testiranje videa

    print("Završeno testiranje!")


def meni_4():
    print("Testiranje zvuka počelo...")

    # todo zovni funkciju za testiranje zvuka

    print("Završeno testiranje!")


def meni_5():
    record_video()
    # todo obradi ovo sto se snimilo i prikazi rezultat


def meni_6():
    captured_audio = record_voice()
    save_recording(captured_audio)
    # todo obradi ovo sto se snimilo i prikazi rezultat


if __name__ == '__main__':
    meni()
